import os


choose = input("delete all train data? (y/n): ")

if choose.lower() == "y":
    for folder_name in ["Accs", "Costs", "Kappas", "Marco_f_measures",
                        "Marco_precisions", "Marco_recalls", "Micro_f_measures"]:
        for file_name in os.listdir("./Log/" + "/" + folder_name):
            os.remove("./Log" + "/" + folder_name + "/" + file_name)

    for file_name in os.listdir("./Model"):
        os.remove("./Model/" + file_name)

    with open("./log.txt", "w") as file:
        file.write("")

    print("ok")
else:
    print("abort.")
