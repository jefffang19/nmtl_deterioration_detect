csv_1_path = "yolov5s_clean_new_predictions.csv"
csv_2_path = "concat_bestand4.csv"

cls_1 = [2]
cls_2 = [1, 3, 4, 5]

output = ['image_filename,label_id,x,y,w,h,confidence\n']

f1 = open(csv_1_path)
f1_lines = f1.readlines()

label_pos = 1
if f1_lines[0][0] == ",":
    label_pos = 2

for i in f1_lines[1:]:
    if int(i.split(',')[label_pos]) in cls_1:
        output.append("{},{},{},{},{},{},{}".format(
            *i.split(',')[label_pos-1:]))

f1.close()

f1 = open(csv_2_path)
f1_lines = f1.readlines()

label_pos = 1
if f1_lines[0][0] == ",":
    label_pos = 2

for i in f1_lines[1:]:
    if int(i.split(',')[label_pos]) in cls_2:
        output.append("{},{},{},{},{},{},{}".format(
            *i.split(',')[label_pos-1:]))

f1.close()

output = [output[0]] + sorted(output[1:])

f1 = open("concat.csv", "w")

for i in output:
    f1.write(i)

f1.close()
