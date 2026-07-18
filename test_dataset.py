from datasets.loveda_dataset import LoveDADataset

dataset = LoveDADataset(
    root_dir="/Users/victoria.zhuravleva/Desktop/учеба/ДИПЛОМ/Летняя практика/semantic_scene_description/semantic_scene_description/data/raw/LoveDA_small",
    split="Train",
    texture="lbp"
)

image, mask = dataset[0]

print(image.shape)
print(mask.shape)