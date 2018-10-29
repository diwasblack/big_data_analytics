precision = [0.98717949, 0.97530864, 1.        , 0.97333333, 0.9625    , 1.        , 0.47852761, 0.70338983, 1.        , 1]
recall = [0.98717949, 0.94047619, 0.43209877, 0.81111111, 0.98717949, 0.9047619 , 1.        , 0.98809524, 0.43209877, 0.96153846]
fscore = [0.98717949, 0.95757576, 0.60344828, 0.88484848, 0.97468354, 0.95      , 0.6473029 , 0.82178218, 0.60344828, 0.98039216]
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


