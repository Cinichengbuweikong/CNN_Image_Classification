from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk
from tkinter.messagebox import *












def show_window():
    image_path = None

    window = Tk()
    status = Text(window)
    image_widget = Label(window)

    def start_action():
        nonlocal image_path
        nonlocal image_widget
        nonlocal status

        image_path = tkinter.filedialog.askopenfilename()
        image_path = image_path.replace("/", "\\\\")
        status.insert("end", f"选择文件: {image_path}\n")

        img_open = Image.open(image_path)
        img_open = img_open.resize((200, 200))
        img_png = ImageTk.PhotoImage(img_open)
        image_widget.config(image=img_png)
        image_widget.image = img_png


        # 准备数据
        status.insert("end", f"读取数据\n")

        data_x = np.empty((0, 256, 256, 3))
        data_y = np.empty((0, 5))

        value = img_path.split(".")[-2]
        value = value.split("/")[-1]
        value = target_to_vector(value)
        value = value[np.newaxis, :]

        img = Image.open(image_path)
        img = np.asarray(img)
        img = img[np.newaxis, :, :, :]

        data_x = np.concatenate([data_x, img])
        data_y = np.concatenate([data_y, value])


        # 开始预测
        status.insert("end", f"开始预测\n")
        pred, real, acc = test(data_x, data_y)

        status.insert("end", f"预测结果: 预测: {pred}  实际: {real} \n")

        # showinfo("预测结果", f"预测: {pred}  实际: {real}")
        print(pred, real)


    window.title("Predict")

    status.pack()
    status.insert("end", "等待选择文件...\n")

    Button(window, text="路径选择", command=start_action).pack()

    image_widget.pack()

    def update_window():
        window.update()
        window.after(500, update_window)

    window.after(500, update_window)

    window.mainloop()


show_window()
