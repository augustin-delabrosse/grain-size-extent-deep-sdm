## Train test val split
train_test_split_output_dir = '../../donnees_terrain/donnees_modifiees/train_test_split/'

train_patches_output_dir = train_test_split_output_dir + "train_patches/"
test_patches_output_dir = train_test_split_output_dir + "test_patches/"
val_patches_output_dir = train_test_split_output_dir + "val_patches/"


## Labels
labels_path = "../labels_rsec2026.csv"

expected_bands_dict = {
    'S2_MS_patches': [4],  
    'S2_RE_patches': [1],        
    'Drone_MS_patches': [5, 10],
    "SR_MS_patches": [4]
}