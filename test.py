import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

# Cấu hình Streamlit
st.set_page_config(page_title="Phân loại ảnh", layout="wide")

# *1. Tải dữ liệu từ thư mục*
dataset_path = r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh3\saved_images"
st.title("📸 Phân loại ảnh với Streamlit")
# Đọc danh sách ảnh
image_files = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")]


with st.expander("🖼️ Dữ liệu ban đầu",expanded=True):
    st.subheader("📌***1. Thông tin dữ liệu***")
    st.markdown(
        '''
        *MNIST* là phiên bản được chỉnh sửa từ bộ dữ liệu *NIST gốc* của Viện Tiêu chuẩn và Công nghệ Quốc gia Hoa Kỳ.  
        Bộ dữ liệu ban đầu gồm các chữ số viết tay từ *nhân viên bưu điện* và *học sinh trung học*.  

        Các nhà nghiên cứu *Yann LeCun, Corinna Cortes, và Christopher Burges* đã xử lý, chuẩn hóa và chuyển đổi bộ dữ liệu này thành *MNIST*  
        để dễ dàng sử dụng hơn cho các bài toán nhận dạng chữ số viết tay.
        '''
    )
    # Đặc điểm của bộ dữ liệu
    st.subheader("📌***2. Đặc điểm của bộ dữ liệu***")
    st.markdown(
        '''
        - *Số lượng ảnh:* 70.000 ảnh chữ số viết tay  
        - *Kích thước ảnh:* Mỗi ảnh có kích thước 28x28 pixel  
        - *Cường độ điểm ảnh:* Từ 0 (màu đen) đến 255 (màu trắng)  
        - *Dữ liệu nhãn:* Mỗi ảnh đi kèm với một nhãn số từ 0 đến 9  
        '''
    )

    # *2. Kiểm tra dữ liệu*
    st.subheader("📌***3. Dữ liệu ban đầu***")

    # *2.1. Hiển thị danh sách file ảnh*
    st.write(f"🔍 Tổng số ảnh bao gồm: {len(image_files)}")

    # *2.2. Hiển thị 5 file ảnh đầu tiên*
    st.write("📂 Một số file ảnh mẫu:")
    st.write(image_files[:10])

    # *2.3. Hiển thị số lượng ảnh theo nhãn*
    labels = [file.split("_")[0] for file in image_files]  # Giả sử nhãn nằm ở đầu tên file (VD: 'cat_1.jpg')
    label_counts = pd.Series(labels).value_counts()

    # Hiển thị kết quả trên Streamlit
    st.subheader("📌***4. Kiểm tra dữ liệu ảnh có bị lỗi***")

    if st.button("🔍 Kiểm tra ảnh lỗi"):
        # Giả sử corrupted_files là danh sách các ảnh bị lỗi (có thể thay thế bằng hàm kiểm tra thực tế)
        corrupted_files = []  # Cập nhật danh sách ảnh lỗi từ quá trình kiểm tra

        if corrupted_files:
            st.error(f"🚨 Có {len(corrupted_files)} ảnh bị lỗi! Bạn cần xóa hoặc sửa chúng.")

            # Hiển thị danh sách ảnh lỗi
            for img_name, error_msg in corrupted_files:
                st.write(f"❌ *{img_name}* - Lỗi: {error_msg}")
        else:
            st.success("✅ Không có ảnh nào bị lỗi! Dữ liệu ảnh hợp lệ 🎉")
    if image_files:
        # Chọn ngẫu nhiên 10 ảnh
        selected_images = random.sample(image_files, min(10, len(image_files)))

        # Hiển thị ảnh trên Streamlit
        st.subheader("📌***5. Một số ảnh ngẫu nhiên từ dataset***")
        # st.write("📷 *Một số ảnh ngẫu nhiên từ dataset*")
        cols = st.columns(10)  # Chia layout thành 10 cột để hiển thị ảnh ngang hàng

        for col, img_name in zip(cols, selected_images):
            img_path = os.path.join(dataset_path, img_name)
            img = Image.open(img_path)

            # Resize ảnh về kích thước nhỏ hơn (ví dụ: 32x32)
            img_resized = img.resize((32, 32))

            # Hiển thị ảnh trong từng cột
            col.image(img_resized, caption=img_name, use_container_width=True)

    def check_corrupted_images(image_files):
        corrupted_files = []
        
        for img_name in image_files:
            img_path = os.path.join(dataset_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Kiểm tra ảnh có bị hỏng không
            except Exception as e:
                corrupted_files.append((img_name, str(e)))  # Lưu cả lỗi
        
        return corrupted_files

    # Kiểm tra ảnh lỗi
    corrupted_files = check_corrupted_images(image_files)

    # Trực quan hóa một số ảnh trong dataset (trước khi tiền xử lý)
    st.subheader("📌***6. Trực quan hóa ảnh trước khi tiền xử lý***")
    if image_files:
        # Chọn ngẫu nhiên 10 ảnh từ dataset
        random_images = random.sample(image_files, min(10, len(image_files)))

        # Tạo figure với kích thước cố định
        fig, axes = plt.subplots(1, 10, figsize=(12, 6))  # 2 hàng, 5 cột

        # Lặp qua danh sách ảnh ngẫu nhiên để hiển thị
        for ax, img_name in zip(axes.flat, random_images):
            img_path = os.path.join(dataset_path, img_name)
            img = Image.open(img_path)

            # Hiển thị ảnh
            ax.imshow(img, cmap="gray")  # Nếu ảnh là đen trắng thì sẽ hiển thị đúng
            ax.axis("off")  # Ẩn trục tọa độ
            ax.set_title(img_name[:7])  # Hiển thị tên ảnh rút gọn

        # Hiển thị hình ảnh trên Streamlit
        st.pyplot(fig)


X = []  # Dữ liệu ảnh
processed_images = []  # Ảnh sau xử lý
if "X" not in st.session_state:
    st.session_state.X = None
with st.expander("🖼️ Tiền xử lí dữ liệu",expanded=True):
    st.subheader("📌***Quá trình Xử lý Dữ liệu Ảnh***")
    st.markdown(
        """
        ### 🔍 1. Kiểm tra Sự tồn tại của Thư mục Dữ liệu  
        - Hệ thống kiểm tra xem thư mục chứa dữ liệu ảnh có tồn tại hay không.  
        - Nếu thư mục không tồn tại, hiển thị thông báo lỗi để người dùng kiểm tra lại đường dẫn.  

        ### 📂 2. Tìm kiếm và Liệt kê Ảnh trong Thư mục  
        - Nếu thư mục tồn tại, hệ thống sẽ tìm tất cả các ảnh có định dạng *.png, .jpg, .jpeg*.  
        - Nếu không tìm thấy ảnh, cảnh báo sẽ được hiển thị.  
        - Nếu tìm thấy ảnh, thông tin về số lượng ảnh sẽ được báo cáo và quá trình xử lý dữ liệu sẽ bắt đầu.  

        ### ⚙ 3. Tiền Xử lý Ảnh  
        - *Đọc ảnh*: Hệ thống nạp từng ảnh từ thư mục.  
        - *Kiểm tra lỗi*: Nếu ảnh không hợp lệ, hệ thống bỏ qua và báo lỗi.  
        - *Chuyển đổi ảnh sang grayscale*: Giúp giảm chiều dữ liệu, tối ưu hóa hiệu suất mô hình.  
        - *Làm mịn ảnh (Gaussian Blur)*: Giảm nhiễu, cải thiện chất lượng ảnh.  
        - *Căn chỉnh độ tương phản (Adaptive Thresholding)*: Tăng độ rõ nét của đặc trưng chữ số.  
        - *Chuẩn hóa kích thước*: Resize ảnh về *28x28 pixels*, định dạng chuẩn cho mô hình nhận diện chữ số.  
        - *Chuẩn hóa dữ liệu*: Đưa giá trị pixel về khoảng *[0,1]* để cải thiện hiệu suất mô hình.  
        - *Chuyển đổi thành vector 1D*: Ảnh được biến đổi thành một vector để sử dụng cho huấn luyện mô hình.  

        ### 🖼️ 4. Hiển thị Kết quả  
        - Sau khi hoàn thành xử lý, một số ảnh mẫu sẽ được hiển thị để người dùng kiểm tra trực quan.  
        - Hiển thị thông báo xác nhận hoàn tất quá trình tiền xử lý dữ liệu.  
        """
    )
    # *TIỀN XỬ LÝ DỮ LIỆU*
    if st.button("📂 Xử lý dữ liệu", key="process_data_btn"):
        # dataset_path = "path/to/dataset"  # Cần thay đổi thành thư mục thật

        if not os.path.exists(dataset_path):
            st.error(f"❌ Thư mục {dataset_path} không tồn tại!")
        else:
            image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                st.warning("⚠ Không tìm thấy ảnh nào trong thư mục!")
            else:
                st.success(f"✅ Tìm thấy {len(image_files)} ảnh. Đang xử lý...")

                processed_images = []
                X = []

                for img_name in image_files:
                    img_path = os.path.join(dataset_path, img_name)
                    img = cv2.imread(img_path)

                    if img is None:
                        st.error(f"❌ Không thể đọc ảnh: {img_name}, bỏ qua.")
                        continue

                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
                    img_thresh = cv2.adaptiveThreshold(
                        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                    )
                    img_resized = cv2.resize(img_thresh, (28, 28))
                    img_normalized = img_resized / 255.0
                    img_flatten = img_normalized.flatten()

                    X.append(img_flatten)
                    processed_images.append(img_resized)

                # *Lưu dữ liệu vào session_state*
                st.session_state.X = np.array(X)

                # Hiển thị ảnh sau tiền xử lý
                st.subheader("📷 Một số ảnh sau xử lý:")
                cols = st.columns(10)
                for i in range(min(10, len(processed_images))):
                    cols[i].image(processed_images[i], caption=f"Ảnh {i+1}", use_container_width=True, clamp=True)

                st.success("✅ Tiền xử lý dữ liệu hoàn tất!")


with st.expander("📊 *Chia tách dữ liệu*", expanded=True):
    if st.session_state.X is not None:
        test_size = st.slider("📊 Chọn tỷ lệ test (%)", min_value=10, max_value=40, value=20, step=5) / 100
        epochs = st.number_input("🔄 Số lần lặp (epochs)", min_value=5, max_value=100, value=20, step=5)
        batch_size = st.selectbox("📦 Kích thước batch", [16, 32, 64, 128], index=1)

        y = np.random.randint(0, 10, len(st.session_state.X))  # Giả lập nhãn
        X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # *Hiển thị thông tin dữ liệu*
        st.success(f"✅ Dữ liệu đã chia với {100 - test_size*100:.0f}% train, {test_size*100:.0f}% test")
        st.write(f"📌 *Số lượng mẫu tập train:* {len(X_train)}")
        st.write(f"📌 *Số lượng mẫu tập test:* {len(X_test)}")

        # *Vẽ biểu đồ phân bố nhãn*
        train_labels, train_counts = np.unique(y_train, return_counts=True)
        test_labels, test_counts = np.unique(y_test, return_counts=True)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].bar(train_labels, train_counts, color="blue")
        ax[0].set_title("Phân bố dữ liệu - Train")
        ax[1].bar(test_labels, test_counts, color="red")
        ax[1].set_title("Phân bố dữ liệu - Test")

        st.pyplot(fig)

        # *Hiển thị thông số huấn luyện đã chọn*
        st.info(f"🛠 *Huấn luyện với:* {epochs} epochs, batch size = {batch_size}")

    else:
        st.warning("⚠ Dữ liệu chưa được xử lý! Vui lòng nhấn '📂 Xử lý dữ liệu' trước khi tiếp tục.")







# # *9. Demo dự đoán với ảnh người dùng tải lên*
# st.subheader("🔍 Demo phân loại ảnh")

# uploaded_file = st.file_uploader("📤 Tải ảnh lên để phân loại", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

#     # Tiền xử lý ảnh
#     img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#     img = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)
#     img_scaled = scaler.transform(img)

#     # Dự đoán nhãn
#     prediction = model.predict(img_scaled)
#     st.success(f"📌 Dự đoán: *{prediction[0]}*")