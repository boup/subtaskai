import argparse
import json
import numpy as np

from compute_scores import pearson_score
from collaborative_filtering import find_similar_users

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find the Menu recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True,
            help='Input user')
    return parser

# Get Menu recommendations for the input user
def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)

        if similarity_score <= 0:
            continue

        filtered_list = [x for x in dataset[user] if x not in \
                dataset[input_user] or dataset[input_user][x] == 0]

        for item in filtered_list:
            overall_scores.update({item: dataset[user][item] * similarity_score})
            similarity_scores.update({item: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Generate menu ranks by normalization
    menu_scores = np.array([[score/similarity_scores[item], item]
            for item, score in overall_scores.items()])

    # Sort in decreasing order
    menu_scores = menu_scores[np.argsort(menu_scores[:, 0])[::-1]]

    # Extract the menu recommendations
    menu_recommendations = [menu for _, menu in menu_scores]

    return menu_recommendations

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    print("\nMenu recommendations for " + user + ":")
    menus = get_recommendations(data, user)
    for i, menu in enumerate(menus):
        print(str(i+1) + '. ' + menu)
