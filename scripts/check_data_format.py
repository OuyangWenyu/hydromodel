import argparse

from hydromodel.datasets.data_preprocess import check_tsdata_format


def main():
    parser = argparse.ArgumentParser(
        description="Check the format of hydrological data."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the hydrological data file",
    )

    args = parser.parse_args()
    file_path = args.data_file

    if check_tsdata_format(file_path):
        print("Data format is correct.")
    else:
        print("Data format is incorrect.")


if __name__ == "__main__":
    main()
