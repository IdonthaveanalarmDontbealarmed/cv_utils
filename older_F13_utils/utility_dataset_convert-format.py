# Convert a dataset from one format to another (using Ultralytics framework)

from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir='..\\datasets\\data1\\',
    save_dir='..\\datasets\\data1yolo\\',
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False,
)