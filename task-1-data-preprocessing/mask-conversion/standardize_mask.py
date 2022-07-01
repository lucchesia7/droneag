import cv2
import numpy as np
import typer

from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm
from matplotlib import pyplot as plt

# FOLDER_PATH = "../Annotations_ Segmentation Masks/segmentation mask_ batch-1/SegmentationClass"
# LABELMAP = "../Annotations_ Segmentation Masks/segmentation mask_ batch-1/SegmentationClass"
app = typer.Typer()

def imread(filename:Path)->np.array:
    """Read the image into RGB format

    Args:
        filename (Path): file name

    Returns:
        np.array: an image with RGB format
    """

    img = cv2.imread(str(filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_labels(
    labelmap_file: Path
    )->Tuple[
        List[np.array],
        List[str]
        ]:
    """Get the labels information

    Args:
        labelmap_file (Path): the labelmap filename
            that consists of the following:

            # label:color_rgb:parts:actions
            label_0:r0,g0,b0::
            label_1:r1,g1,b1::
            label_2:r2,g2,b2::
            .
            .

            where each labels has its own value of r, g, and b

    Returns:
        Tuple[ List[np.array], List[str] ]:
            The RGB value of the labels and 
            the name of the labels

            e.g.: (
                    [[r0, g0, b0],
                    [r1, g1, b1],
                    ...],
                    [label_0, label_1, ....]
                  )
                  
    """

    with open(labelmap_file, 'r') as f:
        data = f.readlines()

    labels = []
    labels_name = []
    for dat in data[1:]:
        dat = dat.split(":")

        labels_name.append(dat[0])
        pixel_label = [
                        int(pixel) 
                        for pixel in dat[1].split(",")
                      ]
        labels.append(np.array(pixel_label))
    
    return (labels, labels_name)

def get_mask(
    img: np.array,
    labels: np.array
    )-> np.array:
    """Get the combined mask

    Args:
        img (np.array): The image
        labels (np.array): The pixel what represents labels

    Returns:
        np.array: The mask
    """

    mask = np.zeros(img.shape[:-1])

    for label, rgb_val in enumerate(labels):
        if label == 0:
            continue
        mask += cv2.inRange(img, rgb_val, rgb_val)/255*label
    
    return mask


def imsave(folder_path:str, mask:np.array, mask_filename:str):
    """Save the image to FOLDER_PATH_segmentation_masks

    Args:
        folder_path (str): The folder path
        mask (np.array): The mask
        mask_filename (str): The mask filename
    """
    save_folder = folder_path+"_segmentation_masks"
    save_folder = Path(folder_path).parent / save_folder
    save_folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_folder / mask_filename), mask)


@app.command()
def run(folder_path: str=typer.Option(
                    "", help="The path of the folder"
                    ),
        label_map: str=typer.Option(
                    "", help="The file of label map"
                    ),):
    mask_files = Path(folder_path).glob("*.png")
    for mask_filename in tqdm(mask_files):
        img = imread(mask_filename)
        labels, _ = get_labels(label_map)
        mask = get_mask(img, labels)
        imsave(folder_path, mask, mask_filename.name)

    # Uncomment to preview the mask sample
    plt.imshow(mask)
    plt.show()

if __name__ == "__main__":
    app()