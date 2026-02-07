import argparse
import hashlib
import json
import io
import os
import tempfile
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Set

from diffusers import AutoencoderOobleck
import torchaudio
import torch


def pad_wav(waveform, segment_length: int):
    waveform_length = len(waveform)
    if waveform_length == segment_length:
        return waveform
    if waveform_length > segment_length:
        return waveform[:segment_length]
    return torch.cat([waveform, torch.zeros(segment_length - waveform_length, device=waveform.device)])


def _resample_to_44100_and_pad(waveform, sr: int, duration_sec: int):
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
    waveform = resampler(waveform)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    target_length = int(44100 * duration_sec)
    padded_left = pad_wav(waveform[0], target_length)
    padded_right = pad_wav(waveform[1], target_length)
    return torch.stack([padded_left, padded_right])


def read_wav_file(filename: str, duration_sec: int):
    info = torchaudio.info(filename)
    sample_rate = info.sample_rate
    num_frames = int(sample_rate * duration_sec)
    waveform, sr = torchaudio.load(filename, num_frames=num_frames)
    return _resample_to_44100_and_pad(waveform, sr, duration_sec)


def read_wav_bytes(flac_bytes: bytes, duration_sec: int):
    fileobj = io.BytesIO(flac_bytes)
    try:
        try:
            info = torchaudio.info(fileobj)
            sample_rate = info.sample_rate
            num_frames = int(sample_rate * duration_sec)
            fileobj.seek(0)
            waveform, sr = torchaudio.load(fileobj, num_frames=num_frames)
        except Exception:
            fileobj.seek(0)
            waveform, sr = torchaudio.load(fileobj)
            waveform = waveform[:, : int(sr * duration_sec)]
        return _resample_to_44100_and_pad(waveform, sr, duration_sec)
    except Exception:
        fileobj.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=True) as tmp:
            tmp.write(fileobj.read())
            tmp.flush()
            return read_wav_file(tmp.name, duration_sec)


def stable_is_val(example_id: str, val_ratio: float) -> bool:
    if val_ratio <= 0:
        return False
    if val_ratio >= 1:
        return True
    h = hashlib.md5(example_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return bucket < val_ratio


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _audio_obj_to_payload(audio_obj: Any) -> Any:
    """Normalize pyarrow to_pylist outputs for HF Audio.

    Expected forms:
      - dict: {'bytes': b'...', 'path': '...', ...}
      - bytes / bytearray / memoryview
      - str (path)
    """
    if isinstance(audio_obj, dict):
        return audio_obj
    if isinstance(audio_obj, (bytes, bytearray, memoryview)):
        return bytes(audio_obj)
    if isinstance(audio_obj, str):
        return audio_obj
    raise TypeError(f"Unexpected audio object type: {type(audio_obj)}")


def _payload_to_audio_bytes_or_path(payload: Any) -> Tuple[Optional[bytes], Optional[str]]:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload), None
    if isinstance(payload, str):
        return None, payload
    if isinstance(payload, dict):
        b = payload.get("bytes")
        p = payload.get("path")
        if b is not None:
            return bytes(b), None
        if p:
            return None, str(p)
        raise ValueError("Audio dict has neither 'bytes' nor 'path'")
    raise TypeError(f"Unexpected payload type: {type(payload)}")


def _extract_prompt_fields(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (prompt, alt_prompt, split).

    Priority:
      1) direct columns if present
      2) parse row['meta'] as JSON
    """
    prompt = row.get("prompt")
    alt_prompt = row.get("alt_prompt")
    split = row.get("split")

    if prompt is not None and alt_prompt is not None and split is not None:
        return str(prompt), str(alt_prompt), str(split)

    meta = row.get("meta")
    if meta is None:
        return str(prompt or ""), str(alt_prompt or ""), str(split or "train")

    if isinstance(meta, (bytes, bytearray, memoryview)):
        meta = bytes(meta).decode("utf-8")

    if isinstance(meta, str):
        try:
            meta_obj = json.loads(meta)
        except Exception:
            meta_obj = {}
    elif isinstance(meta, dict):
        meta_obj = meta
    else:
        meta_obj = {}

    prompt = prompt if prompt is not None else meta_obj.get("prompt", "")
    alt_prompt = alt_prompt if alt_prompt is not None else meta_obj.get("alt_prompt", "")
    split = split if split is not None else meta_obj.get("split", "train")
    return str(prompt), str(alt_prompt), str(split)


def iter_parquet_rows(parquet_glob: str) -> Iterator[Dict[str, Any]]:
    try:
        import glob
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("需要 pyarrow 来读取 parquet：pip install pyarrow") from e

    paths = sorted(glob.glob(parquet_glob))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {parquet_glob}")

    for parquet_path in paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(batch_size=1024):
            cols = record_batch.schema.names
            needed = [c for c in ["id", "input_flac", "gt_flac", "prompt", "alt_prompt", "split", "meta"] if c in cols]
            arrays = {c: record_batch.column(c).to_pylist() for c in needed}
            n = record_batch.num_rows
            for i in range(n):
                yield {c: arrays[c][i] for c in needed}


def _get_dist_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return rank, world_size, local_rank


def _file_barrier(dir_path: str, tag: str, rank: int, world_size: int, timeout_sec: int = 3600) -> None:
    if world_size <= 1:
        return

    _ensure_dir(dir_path)
    marker = os.path.join(dir_path, f".{tag}.rank{rank:03d}")
    done = os.path.join(dir_path, f".{tag}.done")

    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")

    start = time.time()
    if rank == 0:
        while True:
            if all(os.path.exists(os.path.join(dir_path, f".{tag}.rank{r:03d}")) for r in range(world_size)):
                with open(done, "w", encoding="utf-8") as f:
                    f.write("ok\n")
                return
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Barrier timeout waiting for ranks (tag={tag})")
            time.sleep(0.5)


def _load_id_filter(ids_file: Optional[str]) -> Optional[Set[str]]:
    if not ids_file:
        return None
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"ids_file not found: {ids_file}")
    ids: Set[str] = set()
    with open(ids_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "id" in obj:
                        ids.add(str(obj["id"]))
                        continue
                except Exception:
                    pass
            ids.add(line)
    return ids



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SonicMaster .pt latents + train/val/test jsonl from HF parquet Audio columns."
    )
    parser.add_argument("--parquet_glob", type=str, default="/inspire/hdd/global_user/chenxie-25019/HaoQiu/DATA_AND_CKPT/SonicMasterDataset/data/*.parquet")
    parser.add_argument("--out_dir", type=str, default="/inspire/hdd/global_user/chenxie-25019/HaoQiu/DATA_AND_CKPT/FINAL_DATA2")
    parser.add_argument("--input_latents_dir", type=str, default="/inspire/hdd/global_user/chenxie-25019/HaoQiu/DATA_AND_CKPT/FINAL_DATA2/latents_input")
    parser.add_argument("--gt_latents_dir", type=str, default="/inspire/hdd/global_user/chenxie-25019/HaoQiu/DATA_AND_CKPT/FINAL_DATA2/latents_gt")
    parser.add_argument("--train_jsonl", type=str, default="/inspire/hdd/global_user/chenxie-25019/HaoQiu/DATA_AND_CKPT/FINAL_DATA2/trainset_pt.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="/inspire/hdd/global_user/chenxie-25019/HaoQiu/DATA_AND_CKPT/FINAL_DATA2/valset_pt.jsonl")
    parser.add_argument("--test_jsonl", type=str, default="/inspire/hdd/global_user/chenxie-25019/HaoQiu/DATA_AND_CKPT/FINAL_DATA2/testset_pt.jsonl")
    parser.add_argument("--duration_sec", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--max_examples", type=int, default=-1, help="Limit number of assigned examples (rank-local).")
    parser.add_argument("--skip_examples", type=int, default=0, help="Skip N assigned examples (rank-local).")
    parser.add_argument(
        "--ids_file",
        type=str,
        default="",
        help="Optional file containing ids to process (one per line or jsonl with {id}).",
    )
    parser.add_argument("--absolute_paths", action="store_true")
    parser.add_argument(
        "--write_jsonl_live",
        action="store_true",
        help="Append to train/val/test jsonl during processing (single-process only).",
    )
    parser.add_argument(
        "--ignore_dist_env",
        action="store_true",
        help="Ignore RANK/WORLD_SIZE/LOCAL_RANK env vars and run as single process.",
    )
    parser.add_argument(
        "--flush_jsonl",
        action="store_true",
        help="Flush jsonl shard files after each write (useful to observe progress during long runs).",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=200,
        help="Log progress every N assigned examples (rank-local).",
    )
    args = parser.parse_args()

    if args.ignore_dist_env:
        rank, world_size, local_rank = 0, 1, 0
    else:
        rank, world_size, local_rank = _get_dist_info()

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    is_main_process = rank == 0

    _ensure_dir(args.out_dir)
    _ensure_dir(args.input_latents_dir)
    _ensure_dir(args.gt_latents_dir)
    _ensure_dir(os.path.join(args.out_dir, "shards"))
    shard_dir = os.path.join(args.out_dir, "shards")
    _file_barrier(shard_dir, "mkdir", rank=rank, world_size=world_size)

    id_filter = _load_id_filter(args.ids_file) if args.ids_file else None

    if args.write_jsonl_live and world_size != 1:
        raise RuntimeError("--write_jsonl_live 仅支持单进程运行（WORLD_SIZE=1）")

    vae = AutoencoderOobleck.from_pretrained("/inspire/hdd/global_user/chenxie-25019/HaoQiu/DATA_AND_CKPT/stable-audio-open-1.0/vae")
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)

    train_shard = os.path.join(shard_dir, f"train.rank{rank:03d}.jsonl")
    val_shard = os.path.join(shard_dir, f"val.rank{rank:03d}.jsonl")
    test_shard = os.path.join(shard_dir, f"test.rank{rank:03d}.jsonl")

    # Use line-buffering so that shard jsonl becomes visible immediately.
    ft = open(train_shard, "w", encoding="utf-8", buffering=1)
    fv = open(val_shard, "w", encoding="utf-8", buffering=1)
    fte = open(test_shard, "w", encoding="utf-8", buffering=1)

    live_train = live_val = live_test = None
    if args.write_jsonl_live and is_main_process:
        live_train = open(args.train_jsonl, "a", encoding="utf-8", buffering=1)
        live_val = open(args.val_jsonl, "a", encoding="utf-8", buffering=1)
        live_test = open(args.test_jsonl, "a", encoding="utf-8", buffering=1)

    def pick_split(example_id: str, split_value: str) -> str:
        s = (split_value or "train").lower()
        if s in {"test", "testing"}:
            return "test"
        if s in {"val", "valid", "validation", "dev"}:
            return "val"
        return "val" if stable_is_val(example_id, args.val_ratio) else "train"

    batch_inp = []
    batch_gt = []
    batch_meta = []
    global_idx = 0

    # Simple counters for debugging.
    n_seen = 0
    n_assigned = 0
    n_written_train = 0
    n_written_val = 0
    n_written_test = 0
    n_errors = 0

    def flush_batch() -> None:
        nonlocal batch_inp, batch_gt, batch_meta
        nonlocal n_written_train, n_written_val, n_written_test
        if not batch_inp:
            return

        inp_tensor = torch.stack(batch_inp).to(device)
        gt_tensor = torch.stack(batch_gt).to(device)

        with torch.no_grad():
            inp_latents = vae.encode(inp_tensor).latent_dist.mode().transpose(1, 2)
            gt_latents = vae.encode(gt_tensor).latent_dist.mode().transpose(1, 2)

        for (ex_id, prompt, alt_prompt, split_name), inp_lat, gt_lat in zip(batch_meta, inp_latents.cpu(), gt_latents.cpu()):
            inp_path = os.path.join(args.input_latents_dir, f"{ex_id}.pt")
            gt_path = os.path.join(args.gt_latents_dir, f"{ex_id}.pt")
            torch.save(inp_lat, inp_path)
            torch.save(gt_lat, gt_path)

            out_inp_path = os.path.abspath(inp_path) if args.absolute_paths else inp_path
            out_gt_path = os.path.abspath(gt_path) if args.absolute_paths else gt_path

            entry = {
                "id": ex_id,
                "prompt": prompt,
                "alt_prompt": alt_prompt,
                "duration": int(args.duration_sec),
                "location": out_inp_path,
                "original_location": out_gt_path,
                "split": split_name,
            }

            if split_name == "test":
                fte.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written_test += 1
                if args.flush_jsonl:
                    fte.flush()
                if live_test is not None:
                    live_test.write(json.dumps(entry, ensure_ascii=False) + "\n")
            elif split_name == "val":
                fv.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written_val += 1
                if args.flush_jsonl:
                    fv.flush()
                if live_val is not None:
                    live_val.write(json.dumps(entry, ensure_ascii=False) + "\n")
            else:
                ft.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written_train += 1
                if args.flush_jsonl:
                    ft.flush()
                if live_train is not None:
                    live_train.write(json.dumps(entry, ensure_ascii=False) + "\n")

        batch_inp = []
        batch_gt = []
        batch_meta = []

    from tqdm import tqdm

    rows = iter_parquet_rows(args.parquet_glob)
    rows = tqdm(rows, disable=not is_main_process, desc=f"Rank {rank} preparing")

    for row in rows:
        n_seen += 1
        if (global_idx % world_size) != rank:
            global_idx += 1
            continue
        global_idx += 1
        n_assigned += 1

        if args.skip_examples > 0 and n_assigned <= args.skip_examples:
            continue
        if args.max_examples > 0 and (n_assigned - max(args.skip_examples, 0)) > args.max_examples:
            break

        ex_id = row.get("id")
        if ex_id is None:
            continue
        ex_id = str(ex_id)

        if id_filter is not None and ex_id not in id_filter:
            continue

        if "input_flac" not in row or "gt_flac" not in row:
            raise KeyError("Parquet rows must contain input_flac and gt_flac")

        inp_payload = _audio_obj_to_payload(row["input_flac"])
        gt_payload = _audio_obj_to_payload(row["gt_flac"])

        inp_bytes, inp_path = _payload_to_audio_bytes_or_path(inp_payload)
        gt_bytes, gt_path = _payload_to_audio_bytes_or_path(gt_payload)

        try:
            inp_wav = read_wav_bytes(inp_bytes, args.duration_sec) if inp_bytes is not None else read_wav_file(inp_path, args.duration_sec)  # type: ignore[arg-type]
            gt_wav = read_wav_bytes(gt_bytes, args.duration_sec) if gt_bytes is not None else read_wav_file(gt_path, args.duration_sec)  # type: ignore[arg-type]

            prompt, alt_prompt, split_value = _extract_prompt_fields(row)
            split_name = pick_split(ex_id, split_value)

            batch_inp.append(inp_wav)
            batch_gt.append(gt_wav)
            batch_meta.append((ex_id, prompt, alt_prompt, split_name))

            if len(batch_inp) >= args.batch_size:
                flush_batch()
        except Exception as e:
            n_errors += 1
            if is_main_process:
                print(f"Error on id={ex_id}: {e}")

        if args.log_every > 0 and (n_assigned % args.log_every) == 0 and is_main_process:
            print(
                f"[rank{rank}] seen={n_seen} assigned={n_assigned} "
                f"written(train/val/test)={n_written_train}/{n_written_val}/{n_written_test} errors={n_errors}"
            )

    flush_batch()

    ft.close()
    fv.close()
    fte.close()
    if live_train is not None:
        live_train.close()
    if live_val is not None:
        live_val.close()
    if live_test is not None:
        live_test.close()

    done_marker = os.path.join(shard_dir, f".encode_done.rank{rank:03d}")
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    _file_barrier(shard_dir, "encode_done", rank=rank, world_size=world_size)

    if is_main_process:
        print(
            f"[rank{rank}] FINAL seen={n_seen} assigned={n_assigned} "
            f"written(train/val/test)={n_written_train}/{n_written_val}/{n_written_test} errors={n_errors}"
        )
        def merge(out_path: str, pattern: str) -> None:
            _ensure_dir(os.path.dirname(out_path) or ".")
            with open(out_path, "w", encoding="utf-8") as fout:
                for r in range(world_size):
                    shard_path = os.path.join(shard_dir, pattern.format(rank=r))
                    if not os.path.exists(shard_path):
                        continue
                    with open(shard_path, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)

        merge(args.train_jsonl, "train.rank{rank:03d}.jsonl")
        merge(args.val_jsonl, "val.rank{rank:03d}.jsonl")
        merge(args.test_jsonl, "test.rank{rank:03d}.jsonl")

        print("Done. Wrote:")
        print(f"  {args.train_jsonl}")
        print(f"  {args.val_jsonl}")
        print(f"  {args.test_jsonl}")


if __name__ == "__main__":
    main()
