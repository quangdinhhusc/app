import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# Tải dữ liệu MNIST từ OpenML
def data():
    st.header("MNIST Dataset")
    st.write("""
      **MNIST** là một trong những bộ dữ liệu nổi tiếng và phổ biến nhất trong cộng đồng học máy, 
      đặc biệt là trong các nghiên cứu về nhận diện mẫu và phân loại hình ảnh.
  
      - Bộ dữ liệu bao gồm tổng cộng **70.000 ảnh chữ số viết tay** từ **0** đến **9**, 
        mỗi ảnh có kích thước **28 x 28 pixel**.
      - Chia thành:
        - **Training set**: 60.000 ảnh để huấn luyện.
        - **Test set**: 10.000 ảnh để kiểm tra.
      - Mỗi hình ảnh là một chữ số viết tay, được chuẩn hóa và chuyển thành dạng grayscale (đen trắng).
  
      Dữ liệu này được sử dụng rộng rãi để xây dựng các mô hình nhận diện chữ số.
      """)

    st.subheader("Một số hình ảnh từ MNIST Dataset")
    st.image("buoi4/img3.png", caption="Một số hình ảnh từ MNIST Dataset", use_container_width=True)

    st.subheader("Ứng dụng thực tế của MNIST")
    st.write("""
      Bộ dữ liệu MNIST đã được sử dụng trong nhiều ứng dụng nhận dạng chữ số viết tay, chẳng hạn như:
      - Nhận diện số trên các hoá đơn thanh toán, biên lai cửa hàng.
      - Xử lý chữ số trên các bưu kiện gửi qua bưu điện.
      - Ứng dụng trong các hệ thống nhận diện tài liệu tự động.
    """)

    st.subheader("Ví dụ về các mô hình học máy với MNIST")
    st.write("""
      Các mô hình học máy phổ biến đã được huấn luyện với bộ dữ liệu MNIST bao gồm:
      - **Logistic Regression**
      - **Decision Trees**
      - **K-Nearest Neighbors (KNN)**
      - **Support Vector Machines (SVM)**
      - **Convolutional Neural Networks (CNNs)**
    """)

    st.subheader("Kết quả của một số mô hình trên MNIST ")
    st.write("""
      Để đánh giá hiệu quả của các mô hình học máy với MNIST, người ta thường sử dụng độ chính xác (accuracy) trên tập test:
      
      - **Decision Tree**: 0.8574
      - **SVM (Linear)**: 0.9253
      - **SVM (poly)**: 0.9774
      - **SVM (sigmoid)**: 0.7656
      - **SVM (rbf)**: 0.9823
      
      
      
    """)



# Hàm vẽ biểu đồ
def split_data():
    
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    X = np.load("buoi4/X.npy")
    y = np.load("buoi4/y.npy")
    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:", 1000, total_samples, 10000)

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn tỷ lệ test:", 0.1, 0.5, 0.2)

    if st.button("✅ Xác nhận & Lưu"):
        # Lấy số lượng ảnh mong muốn
        X_selected, y_selected = X[:num_samples], y[:num_samples]

        # Chia train/test theo tỷ lệ đã chọn
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=test_size, random_state=42)

        # Lưu vào session_state để sử dụng sau
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.success(f"🔹 Dữ liệu đã được chia: Train ({len(X_train)}), Test ({len(X_test)})")

    # Kiểm tra nếu đã lưu dữ liệu vào session_state
    if "X_train" in st.session_state:
        st.write("📌 Dữ liệu train/test đã sẵn sàng để sử dụng!")
        
def train():
    # 📥 **Tải dữ liệu MNIST**
    if "X_train" in st.session_state:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
    else:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    # 🌟 Chuẩn hóa dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    st.header("⚙️ Chọn mô hình & Huấn luyện")

    # 📌 **Chọn mô hình**
    model_choice = st.selectbox("Chọn mô hình:", ["K-Means", "DBSCAN"])

    if model_choice == "K-Means":
        st.markdown("""
        - **🔹 K-Means** là thuật toán phân cụm phổ biến, chia dữ liệu thành K cụm dựa trên khoảng cách.
        - **Tham số cần chọn:**  
            - **n_clusters**: Số lượng cụm (k).  
        """)
        
        n_clusters = st.slider("n_clusters", 2, 20, 10)
        model = KMeans(n_clusters=n_clusters, random_state=42)
    
    elif model_choice == "DBSCAN":
        st.markdown("""
        - **🛠️ DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là thuật toán phân cụm dựa trên mật độ.
        - **Tham số cần chọn:**  
            - **eps**: Bán kính lân cận.  
            - **min_samples**: Số lượng điểm tối thiểu để tạo cụm.  
        """)
        eps = st.slider("eps", 0.1, 10.0, 0.5)
        min_samples = st.slider("min_samples", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    if st.button("Huấn luyện mô hình"):
        model.fit(X_train)
        labels = model.labels_
        st.success("✅ Huấn luyện thành công!")

        # Lưu mô hình vào session_state dưới dạng danh sách nếu chưa có
        if "models" not in st.session_state:
            st.session_state["models"] = []

        model_name = model_choice.lower().replace(" ", "_")

        existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)
        
        if existing_model:
            count = 1
            new_model_name = f"{model_name}_{count}"
            while any(item["name"] == new_model_name for item in st.session_state["models"]):
                count += 1
                new_model_name = f"{model_name}_{count}"
            model_name = new_model_name
            st.warning(f"⚠️ Mô hình được lưu với tên là: {model_name}")

        st.session_state["models"].append({"name": model_name, "model": model})
        st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        st.write("📋 Danh sách các mô hình đã lưu:")
        model_names = [model["name"] for model in st.session_state["models"]]
        st.write(", ".join(model_names))
        


def ClusteringAlgorithms():
  

    st.title("🖊️ MNIST Classification App")

    ### **Phần 1: Hiển thị dữ liệu MNIST**
    
    ### **Phần 2: Trình bày lý thuyết về Decision Tree & SVM*
    
    # 1️⃣ Phần giới thiệu
    
    # === Sidebar để chọn trang ===
    # === Tạo Tabs ===
    tab1, tab2, tab3, tab4,tab5 = st.tabs(["📘 Lý thuyết Decision Tree", "📘 Lý thuyết SVM", "📘 Data" ,"⚙️ Huấn luyện", "🔢 Dự đoán"])

    with tab1:
        pass

    with tab2:
        pass
    
    with tab3:
        data()
        
    with tab4:
       # plot_tree_metrics()
        
        
        
        split_data()
        train()
        
    
    with tab5:
        pass  





            
if __name__ == "__main__":
    ClusteringAlgorithms()