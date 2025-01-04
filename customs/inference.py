from mmpretrain import ImageClassificationInferencer

if __name__ == '__main__':
    
    config_file = 'customs/aggregation_configs/baseline1_resnet50_malaria_pa3_7_class.py'  # File config của mô hình
    weights_file = 'work_dirs/experiment_result_pa3_res50_oversample/best_accuracy_top1_epoch_32.pth'  # File weights của mô hình
    show_dir = "./work_dirs/visualize/"
    # Tạo đối tượng inference
    inference = ImageClassificationInferencer(
        model = config_file,   # Đường dẫn đến file cấu hình
        pretrained = weights_file # Đường dẫn đến file weights
    )

    # Đường dẫn đến ảnh cần dự đoán
    # image_path = ['dataset/final_malaria_full_class_classification/test/050/rbc_unparasitized/cell.jpg', 
    #               'dataset/final_malaria_full_class_classification/test/050/rbc_unparasitized/cell4.jpg',
    #               'dataset/final_malaria_full_class_classification/test/050/rbc_parasitized_F_TJ/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/129/rbc_parasitized_F_TJ/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/186/rbc_parasitized_F_TJ/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/344/rbc_parasitized_F_TJ/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/103/rbc_parasitized_F_TA/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/103/rbc_parasitized_F_TA/cell2.jpg',
    #               'dataset/final_malaria_full_class_classification/test/103/rbc_parasitized_F_TA/cell3.jpg',
    #               'dataset/final_malaria_full_class_classification/test/103/rbc_parasitized_F_TA/cell4.jpg',
    #               'dataset/final_malaria_full_class_classification/test/222/rbc_parasitized_F_G2-5/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/203/rbc_parasitized_F_G1/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/203/rbc_parasitized_F_G1/cell2.jpg',
    #               'dataset/final_malaria_full_class_classification/test/199/rbc_parasitized_F_G2-5/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/195/rbc_parasitized_F_G2-5/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/194/rbc_parasitized_F_G1/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/525/rbc_parasitized_F_S2/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/525/rbc_parasitized_F_S2/cell2.jpg',
    #               'dataset/final_malaria_full_class_classification/test/525/rbc_parasitized_F_S2/cell3.jpg',
    #               'dataset/final_malaria_full_class_classification/test/554/rbc_parasitized_F_S1/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/569/rbc_parasitized_F_S1/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/199/rbc_unparasitized_dead_kernel/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/184/rbc_unparasitized_artefact/cell2.jpg']

    # image_path = ['dataset/final_malaria_full_class_classification/test/050/rbc_unparasitized/cell.jpg', 
    #               'dataset/final_malaria_full_class_classification/test/050/rbc_unparasitized/cell4.jpg']
    
    image_path = ['dataset/final_malaria_full_class_classification/test/050/rbc_parasitized_F_TJ/cell.jpg',
                  'dataset/final_malaria_full_class_classification/test/129/rbc_parasitized_F_TJ/cell.jpg',
                  'dataset/final_malaria_full_class_classification/test/186/rbc_parasitized_F_TJ/cell.jpg',
                  'dataset/final_malaria_full_class_classification/test/344/rbc_parasitized_F_TJ/cell.jpg',
                  'dataset/final_malaria_full_class_classification/test/150/rbc_parasitized_F_TJ/cell5.jpg']
    
    # image_path = ['dataset/final_malaria_full_class_classification/test/103/rbc_parasitized_F_TA/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/103/rbc_parasitized_F_TA/cell2.jpg',
    #               'dataset/final_malaria_full_class_classification/test/103/rbc_parasitized_F_TA/cell3.jpg',
    #               'dataset/final_malaria_full_class_classification/test/103/rbc_parasitized_F_TA/cell4.jpg']
    
    # image_path = ['dataset/final_malaria_full_class_classification/test/222/rbc_parasitized_F_G2-5/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/203/rbc_parasitized_F_G1/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/203/rbc_parasitized_F_G1/cell2.jpg',
    #               'dataset/final_malaria_full_class_classification/test/199/rbc_parasitized_F_G2-5/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/195/rbc_parasitized_F_G2-5/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/194/rbc_parasitized_F_G1/cell.jpg']
    
    # image_path = ['dataset/final_malaria_full_class_classification/test/525/rbc_parasitized_F_S2/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/525/rbc_parasitized_F_S2/cell2.jpg',
    #               'dataset/final_malaria_full_class_classification/test/525/rbc_parasitized_F_S2/cell3.jpg',
    #               'dataset/final_malaria_full_class_classification/test/554/rbc_parasitized_F_S1/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/569/rbc_parasitized_F_S1/cell.jpg']
    
    # image_path = ['dataset/final_malaria_full_class_classification/test/199/rbc_unparasitized_dead_kernel/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/184/rbc_unparasitized_artefact/cell2.jpg']
    
    # image_path = ['dataset/final_malaria_full_class_classification/test/397/rbc_parasitized_F_TJ/cell.jpg',
    #               'dataset/final_malaria_full_class_classification/test/397/rbc_parasitized_F_TJ/cell2.jpg',
    #               'dataset/final_malaria_full_class_classification/test/397/rbc_parasitized_F_TJ/cell3.jpg']
    
    result = inference(image_path, rescale_factor=1/2, show=True)

    # In kết quả dự đoán
    # print(result['pred_scores'])  # Xác suất dự đoán cho các lớp
    # print(result['pred_label'])   # Nhãn dự đoán
    # print(result['pred_class'])   # Tên của class dự đoán