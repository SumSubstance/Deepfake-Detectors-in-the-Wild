import random
import os
import re
import pandas as pd


LFW_SIZE = 13_233
CELEBA_HQ_SIZE = 30_000
FAIRFACE_SIZE = 97_698


def lfw_logic(female_names, male_names, input_dir, output_dir):
    def get_random_pair(input_list, max_attempts=10_000):
        used_pairs = set()
        attempts = 0

        while attempts < max_attempts:
            source, target = random.sample(input_list, 2)
            pair = (source, target)

            if pair not in used_pairs and source.rpartition('_')[0] != target.rpartition('_')[0]:
                used_pairs.add(pair)
                yield pair
                attempts = 0
            else:
                attempts += 1

    def get_logic(gen_male, gen_female):
        while True:
            if random.uniform(0, 1) < prob_female:
                source, target = next(gen_female)
                part = 'female'
            else:
                source, target = next(gen_male)
                part = 'male'

            source_path = f"{input_dir}/{source.rpartition('_')[0]}/{source}"
            target_path = f"{input_dir}/{target.rpartition('_')[0]}/{target}"
            output_path = f"{output_dir}/{part}_{source.rpartition('.')[0]}_TO_{target.rpartition('.')[0]}.jpg"
            yield source_path, target_path, output_path

    with open(female_names, 'r') as file:
        female_list = [name.strip() for name in file.readlines() if name.strip()]

    with open(male_names, 'r') as file:
        male_list = [name.strip() for name in file.readlines() if name.strip()]

    prob_female = len(female_list) / (len(female_list) + len(male_list))
    gen_female = get_random_pair(female_list)
    gen_male = get_random_pair(male_list)
    gen = get_logic(gen_male, gen_female)

    return gen


def celeba_hq_logic(identity_file, input_dir, output_dir):
    def get_random_pair(input_list, image_label_dict, max_attempts=10_000):
        used_pairs = set()
        attempts = 0

        while attempts < max_attempts:
            source, target = random.sample(input_list, 2)
            pair = (source, target)

            if pair not in used_pairs and image_label_dict[source] != image_label_dict[target]:
                used_pairs.add(pair)
                yield pair
                attempts = 0
            else:
                attempts += 1

    def get_logic(gen_male, gen_female):
        while True:
            if random.uniform(0, 1) < prob_female:
                source, target = next(gen_female)
                part = 'female'
            else:
                source, target = next(gen_male)
                part = 'male'

            source_path = f"{input_dir}/{part}/{source}"
            target_path = f"{input_dir}/{part}/{target}"
            output_path = f"{output_dir}/{part}_{source.rpartition('.')[0]}_TO_{target.rpartition('.')[0]}.jpg"
            yield source_path, target_path, output_path

    image_label_dict = {}
    with open(identity_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                image_label_dict[parts[0]] = int(parts[1])

    female_names = f'{input_dir}/female'
    male_names = f'{input_dir}/male'
    female_list = os.listdir(female_names)
    male_list = os.listdir(male_names)

    prob_female = len(female_list) / (len(female_list) + len(male_list))
    gen_female = get_random_pair(female_list, image_label_dict)
    gen_male = get_random_pair(male_list, image_label_dict)
    gen = get_logic(gen_male, gen_female)

    return gen


def fairface_logic(train_csv, val_csv, input_dir, output_dir):
    def get_logic(df, max_attempts=10_000):
        used_pairs = set()
        attempts = 0

        while attempts < max_attempts:
            source = df.sample(1).values[0]
            source_path = source[0]
            source_age = source[1]
            sorce_gender = source[2]
            source_race = source[3]

            same_params = df[(df['age'] == source_age) & (df['gender'] == sorce_gender) &
                             (df['race'] == source_race) & (df['file'] != source_path)].sample(frac=1)
            if len(same_params) == 0:
                attempts += 1
                continue
            for i in range(len(same_params)):
                target_path = same_params.iloc[i].values[0]
                pair = (source_path, target_path)
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    attempts = 0
                    break
                else:
                    attempts += 1
            source_name = re.findall(r"\d+", source_path)[0]
            target_name = re.findall(r"\d+", target_path)[0]
            source_path = f"{input_dir}/{source_path}"
            target_path = f"{input_dir}/{target_path}"
            output_path = f"{output_dir}/{sorce_gender}_{source_race}_Age_{source_age}_" \
                          f"FROM_{source_name}_TO_{target_name}.jpg"
            yield source_path, target_path, output_path

    df = pd.concat([pd.read_csv(f"{input_dir}/{train_csv}"), pd.read_csv(f"{input_dir}/{val_csv}")],
                   ignore_index=True)
    gen = get_logic(df)
    return gen
