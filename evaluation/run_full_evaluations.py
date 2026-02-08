import argparse
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import yaml

from evaluation.extract_aesthetics_scores import run_aesthetics
from evaluation.extract_fad_mass import run_fad
from evaluation.extract_kl_ssim_mass import run_kl_ssim
from evaluation.evaluate_control_multiple_degs_mass import run_control_evaluation


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_folders(cli_value, config_value):
    if cli_value:
        return cli_value
    if config_value is None:
        return []
    if isinstance(config_value, str):
        return [config_value]
    return list(config_value)



def run_full_suite(
    config_path,
    jsonl_path,
    folders,
    summary_dir,
    clean_targets,
    degraded_root,
    clean_embeddings,
    control_excel,
    aesthetic_csv,
    fad_csv,
    kl_output_dir,
    summary_excel,
    skip_control,
    skip_aesthetics,
    skip_fad,
    skip_kl,
    control_skip_excel,
    kl_save_raw,
    clap_ckpt
):
    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    eval_cfg = config.get("evaluation", {})

    jsonl_path = Path(jsonl_path or paths_cfg.get("infer_file"))
    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl file not found: {jsonl_path}")

    final_folders = resolve_folders(folders, eval_cfg.get("folders"))
    if not final_folders:
        raise ValueError("Need at least one evaluation folder")

    clean_targets_path = clean_targets or eval_cfg.get("clean_targets")
    degraded_path = degraded_root or eval_cfg.get("degraded_root")
    clean_embeddings_path = clean_embeddings or eval_cfg.get("clean_embeddings")

    if clean_embeddings_path and not Path(clean_embeddings_path).exists():
        print(f"⚠️ Clean embeddings missing: {clean_embeddings_path}, skipping FAD")
        clean_embeddings_path = None
    if clean_targets_path and not Path(clean_targets_path).exists():
        print(f"⚠️ Clean target folder missing: {clean_targets_path}, skipping KL/SSIM")
        clean_targets_path = None
    if degraded_path and not Path(degraded_path).exists():
        print(f"⚠️ Degraded folder missing: {degraded_path}, skipping control")
        degraded_path = None
    summary_dir = Path(summary_dir or eval_cfg.get("summary_output") or "evaluation/results")
    summary_dir.mkdir(parents=True, exist_ok=True)
    control_excel_path = Path(control_excel) if control_excel else summary_dir / "control_metrics.xlsx"
    aesthetic_csv_path = Path(aesthetic_csv) if aesthetic_csv else summary_dir / "aesthetic_summary.csv"
    fad_csv_path = Path(fad_csv) if fad_csv else summary_dir / "fad_summary.csv"
    kl_output_dir_path = Path(kl_output_dir) if kl_output_dir else summary_dir / "kl_ssim"

    results_by_type = {}
    summary_frames = []

    if not skip_aesthetics:
        aesthetic_rows = run_aesthetics(str(jsonl_path), final_folders)
        if aesthetic_rows:
            df_aesthetic = pd.DataFrame(aesthetic_rows)
            df_aesthetic["evaluation"] = "aesthetics"
            results_by_type["aesthetics"] = df_aesthetic
            summary_frames.append(df_aesthetic)
            aesthetic_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_aesthetic.to_csv(aesthetic_csv_path, index=False)
            print(f"✨ Aesthetic summary saved to {aesthetic_csv_path}")

    if not skip_fad:
        if not clean_embeddings_path:
            print("⚠️ Skipping FAD because clean embeddings path is missing")
        else:
            fad_results = run_fad(
                str(jsonl_path),
                final_folders,
                clean_embeddings_path,
                output_csv=str(fad_csv_path),
                clap_ckpt=clap_ckpt,
            )
            if fad_results:
                df_fad = pd.DataFrame(fad_results)
                df_fad["evaluation"] = "fad"
                results_by_type["fad"] = df_fad
                summary_frames.append(df_fad)
            else:
                print("⚠️ No FAD metrics produced")

    if not skip_kl:
        if not clean_targets_path:
            print("⚠️ Skipping KL/SSIM because clean targets path is missing")
        else:
            kl_summary = run_kl_ssim(
                str(jsonl_path),
                final_folders,
                clean_targets_path,
                output_dir=str(kl_output_dir_path),
                save_raw=kl_save_raw,
            )
            if kl_summary:
                df_kl = pd.DataFrame(kl_summary)
                df_kl["evaluation"] = "kl_ssim"
                results_by_type["kl_ssim"] = df_kl
                summary_frames.append(df_kl)
            else:
                print("⚠️ No KL/SSIM metrics produced")

    if not skip_control:
        control_df = run_control_evaluation(
            str(jsonl_path),
            final_folders,
            clean_targets_root=clean_targets_path,
            degraded_root=degraded_path,
            excel_output=str(control_excel_path) if not control_skip_excel else None,
            save_excel=not control_skip_excel,
        )
        if not control_df.empty:
            control_df["evaluation"] = "control"
            results_by_type["control"] = control_df
            summary_frames.append(control_df)
        else:
            print("⚠️ Control evaluation produced no rows")

    if summary_frames:
        combined = pd.concat(summary_frames, ignore_index=True)
        combined_path = summary_dir / "evaluation_summary_all.csv"
        combined.to_csv(combined_path, index=False)
        print(f"📊 Combined evaluation summary saved to {combined_path}")

        workbook_path = Path(summary_excel) if summary_excel else summary_dir / "evaluation_summary.csv"
        with pd.ExcelWriter(workbook_path) as writer:
            for name, df in results_by_type.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        print(f"📘 Detailed workbook saved to {workbook_path}")
    else:
        print("⚠️ No evaluation results were collected")
    
    if clap_ckpt:
        print(f"✅ CLAP checkpoint used: {clap_ckpt}")


def main():
    parser = argparse.ArgumentParser(description="Run every evaluation script using the same jsonl config")
    parser.add_argument("--config", default="configs/tangoflux_config.yaml")
    parser.add_argument("--jsonl-path", help="Explicit path to the jsonl file")
    parser.add_argument("--folders", nargs="+", help="Evaluation folders (overrides config)")
    parser.add_argument("--summary-dir", help="Directory for aggregated outputs")
    parser.add_argument("--clean-targets", help="Clean audio root for KL/SSIM")
    parser.add_argument("--degraded-root", help="Degraded audio root for control evaluation")
    parser.add_argument("--clean-embeddings", help="Path to the precomputed clean CLAP embeddings")
    parser.add_argument("--control-excel", help="Excel path for control evaluation")
    parser.add_argument("--aesthetic-csv", help="CSV path for aesthetics summary")
    parser.add_argument("--fad-csv", help="CSV path for FAD summary")
    parser.add_argument("--kl-output-dir", help="Directory for KL/SSIM arrays and summary")
    parser.add_argument("--summary-excel", help="Workbook that holds per-type sheets")
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--skip-aesthetics", action="store_true")
    parser.add_argument("--skip-fad", action="store_true")
    parser.add_argument("--skip-kl", action="store_true")
    parser.add_argument("--control-skip-excel", action="store_true", help="Do not write the control Excel file")
    parser.add_argument("--kl-save-raw", action="store_true", help="Persist KL/SSIM raw arrays")
    parser.add_argument(
        "--clap-ckpt",
        default="/inspire/hdd/global_user/chenxie-25019/HaoQiu/Yesterday_Work/EVAL_MODEL/clap-model/pytorch_model.bin",
        help="Local CLAP checkpoint to avoid downloads",
    )
    args = parser.parse_args()

    run_full_suite(
        config_path=Path(args.config),
        jsonl_path=args.jsonl_path,
        folders=args.folders,
        summary_dir=args.summary_dir,
        clean_targets=args.clean_targets,
        degraded_root=args.degraded_root,
        clean_embeddings=args.clean_embeddings,
        control_excel=args.control_excel,
        aesthetic_csv=args.aesthetic_csv,
        fad_csv=args.fad_csv,
        kl_output_dir=args.kl_output_dir,
        summary_excel=args.summary_excel,
        skip_control=args.skip_control,
        skip_aesthetics=args.skip_aesthetics,
        skip_fad=args.skip_fad,
        skip_kl=args.skip_kl,
        control_skip_excel=args.control_skip_excel,
        kl_save_raw=args.kl_save_raw,
        clap_ckpt=args.clap_ckpt,
    )


if __name__ == "__main__":
    main()
