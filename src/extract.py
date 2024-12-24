import pandas as pd
from PIL import Image
import random
import os


def get_data_dir():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    return os.path.join(parent_dir, 'data')


def load_data(file1, file2):
    prob_df = pd.read_csv(file1)
    mapping_df = pd.read_csv(file2)
    return prob_df, mapping_df


def image_exists(image_path):
    return os.path.exists(image_path)


def classify(probabilities):
    if probabilities['P_CS_DEBIASED'] >= 0.65:
        return 'spiral'
    elif probabilities['P_EL_DEBIASED'] >= 0.65:
        return 'elliptical'
    else:
        return 'irregular'


def get_rows(data_file, mapping_file, images, rows=2000):
    data_df, mapping_df = load_data(data_file, mapping_file)

    galaxies = []
    seen_ids = set()

    while len(galaxies) < rows and len(seen_ids) < len(mapping_df):
        rand = mapping_df.sample(n=1).iloc[0]
        objid = rand['objid']
        asset_id = rand['asset_id']

        if objid in seen_ids:
            continue

        prob_row = data_df[data_df['OBJID'] == objid]
        if prob_row.empty:
            seen_ids.add(objid)
            continue

        image_path = os.path.join(images, f"{asset_id}.jpg")
        if not image_exists(image_path):
            seen_ids.add(objid)
            continue

        classification = classify(prob_row.iloc[0])
        seen_ids.add(objid)
        galaxies.append({
            'objid': objid,
            'image_num': asset_id,
            'picture_path': f"clean_data/images/{asset_id}.jpg",
            'classification': classification
        })

    return pd.DataFrame(galaxies)


def rotate(img):
    rotate_angle = random.choice([0, 90, 180, 270])
    return img.rotate(rotate_angle)


def save_new(galaxies_df, images_dir, output_dir='clean_data'):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_output_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)

    csv_path = os.path.join(output_dir, 'galaxies.csv')
    galaxies_df.to_csv(csv_path, index=False)

    for _, row in galaxies_df.iterrows():
        src_image_path = os.path.join(images_dir, f"{row['image_num']}.jpg")
        dst_image_path = os.path.join(images_output_dir, f"{row['image_num']}.jpg")

        img = Image.open(src_image_path)
        img = rotate(img)
        img.save(dst_image_path)


if __name__ == '__main__':
    data_dir = get_data_dir()
    data_file = os.path.join(data_dir, 'GalaxyZoo1_DR_table2.csv')
    mapping_file = os.path.join(data_dir, 'gz2_filename_mapping.csv')
    images_dir = os.path.join(data_dir, 'images_gz2', 'images')

    galaxies_df = get_rows(data_file, mapping_file, images_dir)
    save_new(galaxies_df, images_dir)