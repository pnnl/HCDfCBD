import argparse
from diomics.cli.vi_pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do-train', action='store_true', help="whether to train the model from scratch")
    parser.add_argument('--do-eval', action='store_true', help="whether to evaluate the model")
    parser.add_argument('--data-dir', default = 'data', help="The path to the folder containing your fmeta and edata files")
    parser.add_argument('--fmeta-file-name', default="fmeta.csv", help="name of the file containing the metadata")
    parser.add_argument('--fmeta-sample-names', nargs='*', default=["SampleID_metab", "SampleID_proteomics", "SampleID_lipidpos"], help="names of columns in fmeta.csv that correspond to the sample IDs in each of the edata files")
    parser.add_argument('--fmeta-target-name', default="Virus", help="name of the column in fmeta.csv that contains the target variable")
    parser.add_argument('--edata-filenames', nargs='*', default=["OMICS_ICL104_Metabolomics_YMK.csv", "ICL104_lipids_aligned_for_stats.csv", "ICL104_proteins_luke.csv"], help="filenames of the edata files")
    parser.add_argument('--edata-cnames', nargs='*', default=["Metabolite", "Name", "Protein"], help="names of the columns in the edata files that contain the biomolecule names")
    parser.add_argument('--preprocess', action='store_true', help = "Apply a naive preprocessing step which median normalizes and mean-imputes the data.  Also removes samples which do not exist across all data types.")
    parser.add_argument('--shapley', action='store_true', help="whether to compute SHAPley values and plot the result")
    parser.add_argument('--igrads', action='store_true', help="whether to compute integrated gradients and plot the result")
    parser.add_argument('--output-dir', default="data/output", help="directory to save the output files")
    parser.add_argument('--best-model-path', default="data/best_model.pt", help="path to save the model")
    parser.add_argument('--final-model-path', default="data/final_model.pt", help="path to load the model")
    parser.add_argument('--patience', type=int, default=10, help="number of validation steps to wait for improvement in validation loss before stopping training")
    parser.add_argument('--validation-frequency', type=int, default=100, help="number of training steps between validation steps")
    parser.add_argument('--max-iter', type=int, default=5000, help="maximum number of training iterations")


    args = parser.parse_args()

    event = {
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "data_dir": args.data_dir,
        "fmeta_file_name": args.fmeta_file_name,
        "fmeta_sample_names": args.fmeta_sample_names,
        "fmeta_target_name": args.fmeta_target_name,
        "edata_filenames": args.edata_filenames,
        "edata_cnames": args.edata_cnames,
        "preprocess": args.preprocess,
        "shapley": args.shapley,
        "igrads": args.igrads,
        "output_dir": args.output_dir,
        "best_model_path": args.best_model_path,
        "final_model_path": args.final_model_path,
        "patience": args.patience,
        "validation_frequency": args.validation_frequency,
        "max_iter": args.max_iter
    }

    run_pipeline(event, None)

if __name__ == "__main__":
    main()