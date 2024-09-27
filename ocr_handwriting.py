# python ocr_handwriting.py -i images/umbc_address.png -m handwriting.model
import tkinter as tk
from tkinter import filedialog, Canvas, Scrollbar
from PIL import Image, ImageTk
import cv2
import numpy as np
import imutils
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours

# Khởi tạo biến args
args = {
    "model": "handwriting.model"  # Đường dẫn đến mô hình nhận dạng chữ viết tay
}

image_path = None

# Khởi tạo cửa sổ chọn file và mô hình OCR
def select_image():
    global image_path
    image_path = filedialog.askopenfilename()   
    if image_path:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

def perform_ocr():
    global image_path
    if image_path:
        # Load mô hình OCR
        model_path = args["model"]
        model = load_model(model_path)
        
        # Đọc ảnh và tiền xử lý
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        
        # Tìm contours và sắp xếp theo chiều từ trái sang phải
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]
        
        # Khởi tạo danh sách các ký tự và vòng lặp qua từng contour
        chars = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
                roi = gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                (tH, tW) = thresh.shape
                
                # Thay đổi kích thước ảnh đã ngưỡng hóa
                if tW > tH:
                    thresh = imutils.resize(thresh, width=32)
                else:
                    thresh = imutils.resize(thresh, height=32)
                
                # Lấp đầy ảnh và chuẩn bị để phân loại
                (tH, tW) = thresh.shape
                dX = int(max(0, 32 - tW) / 2.0)
                dY = int(max(0, 32 - tH) / 2.0)
                padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, 
                                            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                padded = cv2.resize(padded, (32, 32))
                
                # Chuẩn bị dữ liệu và dự đoán
                padded = padded.astype("float32") / 255.0
                padded = np.expand_dims(padded, axis=-1)
                chars.append((padded, (x, y, w, h)))
        
        # Lấy các ký tự và hộp bao
        boxes = [b[1] for b in chars]
        chars = np.array([c[0] for c in chars], dtype="float32")
        
        # Dự đoán và hiển thị kết quả
        preds = model.predict(chars)
        labelNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for (pred, (x, y, w, h)) in zip(preds, boxes):
            i = np.argmax(pred)
            prob = pred[i]
            label = labelNames[i]
            print("[INFO] {} - {:.2f}%".format(label, prob * 100))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Hiển thị ảnh kết quả trong giao diện
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

# Khởi tạo giao diện tkinter
root = tk.Tk()
root.title("Nhận dạng chữ viết tay")

# Tạo Frame chứa Canvas và Scrollbar
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Tạo Canvas để hiển thị hình ảnh
canvas = Canvas(frame, width=800, height=600)
canvas.pack(side=tk.LEFT)

# Tạo Scrollbar cho Canvas
scrollbar = Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)

# Tạo nút chọn hình ảnh
select_button = tk.Button(root, text="Chọn Hình Ảnh", command=select_image)
select_button.pack(side=tk.LEFT, padx=10, pady=10)

# Tạo nút thực hiện OCR
ocr_button = tk.Button(root, text="Thực Hiện OCR", command=perform_ocr)
ocr_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Bắt đầu vòng lặp chính của tkinter
root.mainloop()
