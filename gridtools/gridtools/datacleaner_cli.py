import argparse
import os

"""
This module is a wrapper for the datacleaner and its configuration file to make it easily accessible with a terminal interface.
"""


def main():
    parser = argparse.ArgumentParser(
        prog="datacleaner-cli",
        description="Terminal interface to preprocess tracked electrode grid recordings. Edit the yaml-configuration file using the keyword 'edit' and prove a text editor or fallback to the default VSCode or run the preprocessing unsing the keyword 'run'. It is advised to run a dry-run (i.e. without writing to disk) first, to ensure that no errors occur during preprocessing. This function can be enabled and disabled in the configuration file.",
    )

    parser.add_argument(
        "mode",
        metavar="Keyword",
        type=str,
        help="'edit' to edit the configuration file, 'run' to run the datacleaner.",
    )

    parser.add_argument(
        "--editor",
        metavar="Text-editor",
        type=str,
        default="code",
        help="E.g. 'code' to edit in VSCode, 'pycharm' to edit in pycharm, etc.",
    )

    args = parser.parse_args()
    here = os.path.abspath(os.path.dirname(__file__))
    confpath = here + "/datacleaner_conf.yml"

    if args.mode == "run":
        os.system(f"python {here}/datacleaner.py")
    elif args.mode == "edit":
        editor = args.editor

        try:
            os.system(f"{editor} {confpath}")
        except:
            print(
                "Editor not found. Provide a different text editor to edit the configuration file."
            )


if __name__ == "__main__":
    main()
