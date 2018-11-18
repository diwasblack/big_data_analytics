precision = [0.47674419, 0.59459459, 0.77108434, 0.58      , 0.53658537, 0.39759036, 0.49180328, 0.52631579, 0.59302326, 0.43076923]
recall = [0.52564103, 0.52380952, 0.79012346, 0.64444444, 0.28205128, 0.39285714, 0.76923077, 0.47619048, 0.62962963, 0.35897436]
fscore = [0.5       , 0.55696203, 0.7804878 , 0.61052632, 0.3697479 , 0.39520958, 0.6       , 0.5       , 0.61077844, 0.39160839]
support = [78, 84, 81, 90, 78, 84, 78, 84, 81, 78]


labels = [
    "mantled_howler",
    "patas_monkey",
    "bald_uakari",
    "japanese_macaque",
    "pygmy_marmoset",
    "white_headed_capuchin",
    "silvery_marmoset",
    "common_squirrel_monkey",
    "black_headed_night_monkey",
    "nilgiri_langur",
]


if __name__ == "__main__":
    with open("performance.csv", "w") as file:
        file.write("Class, Precision, Recall, Fscore, Support\n")
        for i, item in enumerate(zip(precision, recall, fscore, support)):
            csv_str = "{}, {}, {}, {}, {}\n".format(labels[i], *item)
            file.write(csv_str)
