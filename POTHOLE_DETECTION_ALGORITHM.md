# Algorithm 1: Real-Time Pothole Detection and Mapping System

**Require:** Image stream I, Model checkpoint M, GPS coordinates G, Confidence threshold θ, Input size s, ROS topics T  
**Ensure:** Real-time pothole detection with GPS mapping and severity assessment

```
1:  for each frame Fi ∈ I do
2:      Load frame Fi from camera stream at timestamp ti
3:      preprocessing_data ← ∅
4:      
5:      // Image Preprocessing Pipeline
6:      if frame_valid(Fi) then
7:          Iresize ← resize(Fi, s × s) using bilinear interpolation
8:          Inorm ← normalize(Iresize, μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])
9:          Itensor ← convert_to_tensor(Inorm) with shape [1, 3, s, s]
10:         Add (Itensor, ti, Fi) to preprocessing_data
11:     end if
12:     
13:     // CNN Feature Extraction and Inference
14:     Load tensor X, timestamp t, original_frame F from preprocessing_data
15:     with autocast() do
16:         features ← backbone(X) using depthwise separable convolutions
17:         classification_logits ← classification_head(features)
18:         severity_logits ← severity_head(features) 
19:         depth_prediction ← depth_head(features)
20:     end with
21:     
22:     // Multi-task Prediction Processing
23:     pothole_prob ← softmax(classification_logits)[1]
24:     severity_scores ← softmax(severity_logits)
25:     estimated_depth ← sigmoid(depth_prediction) × max_depth
26:     
27:     // Decision Logic and Filtering
28:     if pothole_prob ≥ θ then
29:         severity_class ← argmax(severity_scores)
30:         confidence_score ← pothole_prob
31:         
32:         // GPS Coordinate Integration
33:         current_gps ← get_gps_coordinates(t)
34:         if gps_valid(current_gps) then
35:             latitude, longitude ← current_gps.lat, current_gps.lon
36:         else
37:             latitude, longitude ← interpolate_gps(t)
38:         end if
39:         
40:         // Pothole Detection Result
41:         detection_result ← {
42:             'timestamp': t,
43:             'confidence': confidence_score,
44:             'severity': severity_class,
45:             'depth_estimate': estimated_depth,
46:             'gps_coordinates': (latitude, longitude),
47:             'image_path': save_detection_image(F, t)
48:         }
49:         
50:         // Database Storage and ROS Publishing
51:         store_detection(detection_result, database)
52:         publish_to_ros(detection_result, T.pothole_detection)
53:         
54:         // Visualization and Logging
55:         annotated_frame ← draw_detection_overlay(F, detection_result)
56:         publish_to_ros(annotated_frame, T.visualization)
57:         log_detection(detection_result)
58:     end if
59:     
60:     // Performance Monitoring
61:     fps ← calculate_fps(t, previous_timestamp)
62:     if fps < min_fps_threshold then
63:         adjust_processing_parameters()
64:     end if
65:     
66: end for
```

---

# Algorithm 2: CNN Training Pipeline for Pothole Detection

**Require:** Dataset D, Model architecture A, Hyperparameters H, Validation split ratio r  
**Ensure:** Trained pothole detection model with optimal weights

```
1:  // Dataset Preparation and Loading
2:  train_data, val_data, test_data ← split_dataset(D, ratios=[0.7, 0.2, 0.1])
3:  transforms ← create_augmentation_pipeline()
4:  train_loader ← DataLoader(train_data, batch_size=H.batch_size, shuffle=True)
5:  val_loader ← DataLoader(val_data, batch_size=H.batch_size, shuffle=False)
6:  
7:  // Model Initialization
8:  model ← create_model(A, num_classes=H.num_classes, task=H.task)
9:  optimizer ← AdamW(model.parameters(), lr=H.learning_rate, weight_decay=H.weight_decay)
10: scheduler ← CosineAnnealingLR(optimizer, T_max=H.epochs)
11: scaler ← GradScaler() for mixed precision training
12: early_stopping ← EarlyStopping(patience=H.patience)
13: 
14: // Training Loop
15: for epoch e ∈ {1, 2, ..., H.epochs} do
16:     model.train()
17:     train_loss ← 0
18:     
19:     for batch (X, y_class, y_severity, y_depth) ∈ train_loader do
20:         optimizer.zero_grad()
21:         
22:         with autocast() do
23:             pred_class, pred_severity, pred_depth ← model(X)
24:             
25:             // Multi-task Loss Computation
26:             loss_class ← CrossEntropyLoss(pred_class, y_class)
27:             loss_severity ← CrossEntropyLoss(pred_severity, y_severity)
28:             loss_depth ← MSELoss(pred_depth, y_depth)
29:             
30:             total_loss ← H.λ₁ × loss_class + H.λ₂ × loss_severity + H.λ₃ × loss_depth
31:         end with
32:         
33:         scaler.scale(total_loss).backward()
34:         scaler.step(optimizer)
35:         scaler.update()
36:         
37:         train_loss ← train_loss + total_loss.item()
38:     end for
39:     
40:     // Validation Phase
41:     model.eval()
42:     val_loss ← 0
43:     val_metrics ← MetricsCalculator()
44:     
45:     with torch.no_grad() do
46:         for batch (X_val, y_val_class, y_val_severity, y_val_depth) ∈ val_loader do
47:             pred_val_class, pred_val_severity, pred_val_depth ← model(X_val)
48:             
49:             val_batch_loss ← compute_validation_loss(pred_val_class, pred_val_severity, pred_val_depth,
50:                                                     y_val_class, y_val_severity, y_val_depth)
51:             val_loss ← val_loss + val_batch_loss
52:             
53:             // Update validation metrics
54:             val_metrics.update(pred_val_class, y_val_class, pred_val_severity, y_val_severity)
55:         end for
56:     end with
57:     
58:     // Learning Rate Scheduling and Early Stopping
59:     scheduler.step()
60:     
61:     if early_stopping(val_loss, model) then
62:         break training loop
63:     end if
64:     
65:     // Model Checkpointing
66:     if val_loss < best_val_loss then
67:         save_checkpoint(model, optimizer, epoch, val_loss, metrics)
68:         best_val_loss ← val_loss
69:     end if
70:     
71: end for
72: 
73: // Final Model Evaluation
74: best_model ← load_best_checkpoint()
75: test_metrics ← evaluate_model(best_model, test_data)
76: save_final_model(best_model, test_metrics)
```

---

# Algorithm 3: ROS-based Multi-Node Pothole Detection System

**Require:** Camera node C, Detection node D, GPS node G, Mapping node M, ROS topics T  
**Ensure:** Distributed real-time pothole detection with mapping integration

```
1:  // Initialize ROS Nodes
2:  rospy.init_node('pothole_detection_system')
3:  
4:  // Camera Node (Node C)
5:  camera ← initialize_usb_camera(device_id=0, resolution=(1920, 1080))
6:  camera.set_properties(fourcc='MJPG', fps=30, auto_exposure=True)
7:  
8:  while rospy.not_shutdown() do
9:      frame ← camera.capture_frame()
10:     if frame_valid(frame) then
11:         timestamp ← rospy.Time.now()
12:         image_msg ← convert_to_ros_image(frame, timestamp)
13:         camera_info_msg ← create_camera_info(frame.shape)
14:         
15:         publish(T.camera_raw, image_msg)
16:         publish(T.camera_info, camera_info_msg)
17:     end if
18: end while
19: 
20: // Detection Node (Node D) 
21: model ← load_trained_model(model_path)
22: subscribe(T.camera_raw, image_callback)
23: 
24: function image_callback(image_msg) do
25:     frame ← convert_from_ros_image(image_msg)
26:     
27:     // Inference Pipeline
28:     preprocessed ← preprocess_image(frame)
29:     with torch.no_grad() do
30:         predictions ← model(preprocessed)
31:         confidence, severity, depth ← parse_predictions(predictions)
32:     end with
33:     
34:     if confidence > detection_threshold then
35:         detection_msg ← create_detection_message(confidence, severity, depth, image_msg.header)
36:         publish(T.pothole_detection, detection_msg)
37:         
38:         // Visualization
39:         annotated_frame ← draw_detection_overlay(frame, predictions)
40:         viz_msg ← convert_to_ros_image(annotated_frame)
41:         publish(T.visualization, viz_msg)
42:     end if
43: end function
44: 
45: // GPS Node (Node G)
46: gps_serial ← initialize_gps_connection(port='/dev/ttyUSB0', baudrate=9600)
47: 
48: while rospy.not_shutdown() do
49:     nmea_data ← gps_serial.read_line()
50:     if validate_nmea(nmea_data) then
51:         gps_fix ← parse_nmea_coordinates(nmea_data)
52:         gps_msg ← create_gps_message(gps_fix)
53:         publish(T.gps_fix, gps_msg)
54:     end if
55: end while
56: 
57: // Mapping Node (Node M)
58: pothole_database ← initialize_sqlite_database()
59: subscribe(T.pothole_detection, pothole_callback)
60: subscribe(T.gps_fix, gps_callback)
61: 
62: function pothole_callback(detection_msg) do
63:     current_gps ← get_latest_gps_fix()
64:     if gps_valid(current_gps) then
65:         pothole_entry ← {
66:             'timestamp': detection_msg.header.stamp,
67:             'latitude': current_gps.latitude,
68:             'longitude': current_gps.longitude,
69:             'confidence': detection_msg.confidence,
70:             'severity': detection_msg.severity,
71:             'depth_estimate': detection_msg.depth
72:         }
73:         
74:         insert_into_database(pothole_database, pothole_entry)
75:         
76:         // Publish mapped pothole
77:         mapped_msg ← create_mapped_pothole_message(pothole_entry)
78:         publish(T.mapped_potholes, mapped_msg)
79:     end if
80: end function
```