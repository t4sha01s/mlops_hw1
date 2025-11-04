import grpc
import app_pb2
import app_pb2_grpc

def test_grpc():
    # Подключаемся к серверу
    channel = grpc.insecure_channel('localhost:50051')
    stub = app_pb2_grpc.MLServiceStub(channel) 
    
    print("=== Testing gRPC Server ===\n")
    
    try:
        # 1. Тест HealthCheck
        print("1. Testing HealthCheck...")
        health_response = stub.HealthCheck(app_pb2.HealthRequest())
        print(f"Success! Status: {health_response.status}")
        
        # 2. Тест GetModelClasses
        print("\n2. Testing GetModelClasses...")
        model_classes = stub.GetModelClasses(app_pb2.Empty())
        for model_name, info in model_classes.model_classes.items():
            print(f"Success! {model_name}: {info.description}")
        
        # 3. Тест TrainModel
        print("\n3. Testing TrainModel...")
        # Создаем FeatureArray для 2D данных
        feature_arrays = []
        for features in [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]]:
            feature_array = app_pb2.FeatureArray()
            feature_array.features.extend(features)
            feature_arrays.append(feature_array)
        
        train_response = stub.TrainModel(app_pb2.TrainRequest(
            model_type="logistic_regression",
            params={"max_iter": "100", "C": "1.0"},
            X=feature_arrays,
            y=[0, 1, 0, 1]
        ))
        model_id = train_response.model_id
        print(f"Success! Model trained: {model_id}")
        print(f"Success! Metrics: {dict(train_response.metrics)}")
        
        # 4. Тест ListModels
        print("\n4. Testing ListModels...")
        models_response = stub.ListModels(app_pb2.Empty())
        print(f"Success! Found {len(models_response.models)} models")
        
        # 5. Тест GetModel
        print("\n5. Testing GetModel...")
        model_info = stub.GetModel(app_pb2.ModelId(model_id=model_id))
        print(f"Success! Model type: {model_info.model_type}")
        print(f"Success! Params: {model_info.params}")
        
        # 6. Тест Predict
        print("\n6. Testing Predict...")
        # Создаем тестовые данные для предсказания
        predict_features = []
        for features in [[1.5, 2.5], [3.5, 4.5]]:
            feature_array = app_pb2.FeatureArray()
            feature_array.features.extend(features)
            predict_features.append(feature_array)
            
        predict_response = stub.Predict(app_pb2.PredictRequest(
            model_id=model_id,
            X=predict_features
        ))
        print(f"Success! Predictions: {predict_response.predictions}")
        
        # 7. Тест GetMetrics
        print("\n7. Testing GetMetrics...")
        metrics_response = stub.GetMetrics(app_pb2.ModelId(model_id=model_id))
        print(f"Success! Metrics: {dict(metrics_response.metrics)}")
        
        # 8. Тест RetrainModel
        print("\n8. Testing RetrainModel...")
        # Новые данные для переобучения
        new_feature_arrays = []
        for features in [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]:
            feature_array = app_pb2.FeatureArray()
            feature_array.features.extend(features)
            new_feature_arrays.append(feature_array)
            
        retrain_response = stub.RetrainModel(app_pb2.RetrainRequest(
            model_id=model_id,
            X=new_feature_arrays,
            y=[1, 0, 1, 0]
        ))
        print(f"Success! Retrained metrics: {dict(retrain_response.metrics)}")
        
        # 9. Тест DeleteModel
        print("\n9. Testing DeleteModel...")
        delete_response = stub.DeleteModel(app_pb2.ModelId(model_id=model_id))
        if delete_response.success:
            print("Success! Model deleted successfully")
        
        print("Success! Success! Success! All gRPC tests passed!")
        
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_grpc()
