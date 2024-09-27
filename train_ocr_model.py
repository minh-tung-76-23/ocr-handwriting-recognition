# python train_ocr_model.py --az a_z_handwritten_data.csv --model handwriting.model

# Đặt các thư viện cần thiết
import matplotlib
matplotlib.use("Agg")  # Sử dụng backend của matplotlib để lưu hình ảnh mà không hiển thị

# Import các thư viện cần thiết
from cnn.models import ResNet  # Import mô hình ResNet từ module cnn.models
from cnn.az_dataset import load_mnist_dataset, load_az_dataset  # Import hàm load dataset từ module cnn.az_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator từ keras để tạo dữ liệu mẫu
from tensorflow.keras.optimizers import SGD  # Import tối ưu hóa SGD từ keras để tối ưu hóa mô hình
from sklearn.preprocessing import LabelBinarizer  # Import LabelBinarizer từ sklearn để mã hoá nhãn
from sklearn.model_selection import train_test_split  # Import train_test_split từ sklearn để chia dữ liệu thành tập huấn luyện và kiểm tra
from sklearn.metrics import classification_report  # Import classification_report từ sklearn để đánh giá mô hình
from imutils import build_montages  # Import build_montages từ imutils để xây dựng montage ảnh
import matplotlib.pyplot as plt  # Import pyplot từ matplotlib để vẽ đồ thị
import numpy as np  # Import numpy để làm việc với mảng đa chiều
import argparse  # Import argparse để phân tích đối số dòng lệnh
import cv2  # Import OpenCV để xử lý ảnh

# Xây dựng bộ phân tích đối số và phân tích các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True,
	help="Đường dẫn đến tập dữ liệu A-Z")  # Đường dẫn đến tập dữ liệu A-Z
ap.add_argument("-m", "--model", type=str, required=True,
	help="Đường dẫn đến mô hình nhận diện chữ viết tay đã huấn luyện")  # Đường dẫn đến mô hình nhận diện chữ viết tay đã huấn luyện
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="Đường dẫn đến tệp lịch sử huấn luyện")  # Đường dẫn đến tệp lịch sử huấn luyện
args = vars(ap.parse_args())  # Phân tích đối số dòng lệnh và lưu vào args

# Khởi tạo số epoch (vòng lặp huấn luyện), learning rate ban đầu và kích thước batch
EPOCHS = 50
INIT_LR = 1e-1
BS = 128

# Tải dữ liệu từ các dataset A-Z và MNIST
print("[INFO] Đang tải các tập dữ liệu...")
(azData, azLabels) = load_az_dataset(args["az"])  # Tải dataset A-Z từ đường dẫn được cung cấp
(digitsData, digitsLabels) = load_mnist_dataset()  # Tải dataset MNIST

# Do dataset MNIST có nhãn từ 0-9, nên ta cộng 10 vào mỗi nhãn của A-Z để đảm bảo các ký tự A-Z không bị gán nhãn nhầm là số
azLabels += 10

# Xếp các dữ liệu và nhãn của A-Z với dữ liệu và nhãn của MNIST
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

# Mỗi ảnh trong dataset A-Z và MNIST đều có kích thước 28x28 pixel;
# Tuy nhiên, kiến trúc mạng mà chúng ta sử dụng yêu cầu ảnh có kích thước 32x32,
# Vì vậy chúng ta cần phải resize chúng lên kích thước 32x32
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

# Thêm một chiều kênh cho mỗi ảnh trong dataset và chuẩn hóa giá trị pixel từ [0, 255] xuống [0, 1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

# Chuyển đổi các nhãn từ dạng số nguyên sang dạng vector
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# Điều chỉnh sự chênh lệch trong dữ liệu đã được gán nhãn
classTotals = labels.sum(axis=0)
classWeight = {}

# Duyệt qua tất cả các lớp và tính toán trọng số của từng lớp
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra, sử dụng 80% dữ liệu cho huấn luyện và 20% còn lại cho kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)

# Xây dựng bộ tạo ảnh để tăng cường dữ liệu
aug = ImageDataGenerator(
	rotation_range=10,  # Góc xoay ảnh trong khoảng [-10, 10]
	zoom_range=0.05,  # Phạm vi zoom ảnh
	width_shift_range=0.1,  # Phạm vi dịch ngang ảnh
	height_shift_range=0.1,  # Phạm vi dịch dọc ảnh
	shear_range=0.15,  # Phạm vi cắt ảnh
	horizontal_flip=False,  # Không lật ảnh theo chiều ngang
	fill_mode="nearest")  # Điền ảnh bằng phương pháp gần nhất

# Khởi tạo và biên dịch mạng nơ-ron sâu của chúng ta
print("[INFO] Đang biên dịch mô hình...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)  # Tối ưu hóa SGD với learning rate và decay
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
	(64, 64, 128, 256), reg=0.0005)  # Xây dựng mô hình ResNet với các tham số tương ứng
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])  # Biên dịch mô hình với hàm loss và optimizer

# Huấn luyện mạng
print("[INFO] Đang huấn luyện mạng...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),  # Dữ liệu được tạo bằng bộ tạo ảnh
	validation_data=(testX, testY),  # Dữ liệu kiểm tra
	steps_per_epoch=len(trainX) // BS,  # Số bước huấn luyện mỗi epoch
	epochs=EPOCHS,  # Số epoch
	class_weight=classWeight,  # Trọng số của các lớp
	verbose=1)  # Chế độ verbose

# Định nghĩa danh sách tên nhãn
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# Đánh giá mạng
print("[INFO] Đang đánh giá mạng...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# Lưu mô hình vào đĩa
print("[INFO] Đang lưu mô hình...")
model.save(args["model"], save_format="h5")

# Xây dựng đồ thị biểu diễn lịch sử huấn luyện
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Biểu đồ Huấn luyện: Loss và Độ chính xác")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Độ chính xác")
plt.legend(loc="lower left")
plt.savefig(args["plot"])  # Lưu đồ thị vào đường dẫn được chỉ định

# Khởi tạo danh sách các hình ảnh đầu ra
images = []

# Ngẫu nhiên chọn một vài ký tự từ tập dữ liệu kiểm tra
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
	# Dự đoán nhãn của ký tự
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labelNames[prediction[0]]

	# Trích xuất ảnh từ dữ liệu kiểm tra và khởi tạo màu văn bản nhãn là màu xanh lá cây (đúng)
	image = (testX[i] * 255).astype("uint8")
	color = (0, 255, 0)

	# Nếu nhãn lớp dự đoán không trùng với nhãn lớp thực sự
	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)  # Đổi màu văn bản nhãn thành màu đỏ

	# Trộn các kênh màu thành một ảnh, resize ảnh từ 32x32 lên 96x96 để nhìn rõ hơn và vẽ nhãn dự đoán lên ảnh
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)  # Vẽ nhãn lên ảnh

	# Thêm ảnh vào danh sách các hình ảnh đầu ra
	images.append(image)

# Xây dựng montage cho các hình ảnh
montage = build_montages(images, (96, 96), (7, 7))[0]

# Hiển thị montage đầu ra
cv2.imshow("Kết quả OCR", montage)
cv2.waitKey(0)
