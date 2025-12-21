import argparse
import sys

from sam3_tools.text_segmentation import run_text_segmentation
from sam3_tools.auto_segmentation import run_auto_segmentation
from sam3_tools.box_segmentation import run_box_segmentation
from sam3_tools.point_segmentation import run_point_segmentation
# from sam3_tools.shared_utils import load_or_create_config, get_config_path


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 segmentation tool")

    parser.add_argument("-i", "--input", required=False, help="Input image path")
    parser.add_argument("-o", "--output", required=False, help="Output folder")
    parser.add_argument("-n", "--num-masks", type=int, default=3, help="Number of masks to save (box and auto mode only)")
    parser.add_argument("-s", "--box", nargs=4, type=int, help="Generate masks from a box selection. Optional box coordinate: x1 y1 x2 y2")
    parser.add_argument("--pfm", action="store_true", help="Save mask as .pfm instead of .png")
    parser.add_argument("--points", action="store_true", help="Generate masks from point-based selection")
    parser.add_argument("--text", type=str, help="Generate masks from text prompt")
    parser.add_argument("--auto", action="store_true", help="Generate automatic masks")
    parser.add_argument("--config", action="store_true", help="Create config file if missing and show the path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Launch GUI if no CLI args were given
    if len(sys.argv) == 1:
        from sam3_tools.gui import start_gui
        start_gui()
        return
    if args.config:
        cfg = load_or_create_config()
        print("Config file is ready at:", get_config_path())
        sys.exit(0)

    # Priority: Text → Points → Auto → Box
    if args.text:
        run_text_segmentation(
            input_path=args.input,
            output_path=args.output,
            prompt=args.text,
            num_masks=args.num_masks,
            pfm=args.pfm,

        )

    elif args.points:
        run_point_segmentation(
            input_path=args.input,
            output_path=args.output,
            num_masks=args.num_masks,
            pfm=args.pfm,
        )

    elif args.auto:
        run_auto_segmentation(
            input_path=args.input,
            output_path=args.output,
            num_masks=args.num_masks,
            pfm=args.pfm,
        )

    else:
        run_box_segmentation(
            input_path=args.input,
            output_path=args.output,
            num_masks=args.num_masks,
            box=args.box,
            pfm=args.pfm,
        )

if __name__ == "__main__":
    main()
