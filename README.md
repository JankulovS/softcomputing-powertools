# softcomputing-powertools


Korišćena Ananconda 3.7 (python 3.6.8) i kreirano virtualno okruženje tf15 sa komandom 'conda create -n tf15 pip'

Potrebne biblioteke:

 - tensorflow-gpu

 - pandas

 - opencv-python

 - pillow
 
 - lxml
 
 - Cython
 
 - jupyter
 
 - matplotlib

  

Testirano na Windows 10

Trenirano na Ryzen 5 1500X, GTX 960 4GB

# Pokretanje već treniranog detektora

Skinuti tensorflow models github repozitorijum sa https://github.com/tensorflow/models i ekstraktovati ga u željeni folder, nakon toga skinuti ovaj repozitorijum i ekstraktovati ga u models/research/object_detection.  

Skinuti folder 'training' sa https://drive.google.com/drive/folders/1GS79qu6LL9NgS32D7HtUZ-FvdqGwj4Ny i postaviti ga u models/research/object_detection, takođe skinuti folder 'inference_graph' i njega postaviti u models/research/object_detection.  

Pokrenuti konzolu i pozicionirati se u conda env tf15 sa komandom 'activate tf15' i nakon toga postaviti PYTHONPATH i PATH sa komandama 
```
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
```
set PATH=%PATH%;PYTHONPATH
```
U slučaju da se putanje foldera razlikuju od iznad naznačenih, promeniti ih u odgovarajuće.  

Uneti komandu 'idle' u konzolu kako bi se otvorio novi python shell, otvoriti preko File->Open skriptu object_detection_image.py i pokrenuti je.  

Ako je sve u redu, nakon desetak sekundi otvoriće se novi prozor sa slikom i označenim alatom. Za drugu ulaznu sliku potrebno je samo promeniti konstantu IMAGE_NAME u odgovarajuću sliku koja se nalazi u folderu skripte i ponovo pokrenuti skriptu, takođe radi na .png formatu.

# Treniranje

Potrebno je skinuti tensorflow models github repozitorijum sa https://github.com/tensorflow/models, nakon čega treba skinuti 'faster_rcnn_v2_coco' sa https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md i ekstraktovati ga u 'research/object_detection' folder. Kreirati folder 'images' i u njemu dva foldera 'test' i 'train'. 

Za treniranje je korišćen dataset https://drive.google.com/drive/folders/1GS79qu6LL9NgS32D7HtUZ-FvdqGwj4Ny?usp=sharing sa ručno anotiranim alatima korišćenjem https://github.com/tzutalin/labelImg alata.  

'test' i 'train' folderi sadrže slike i istoimene xml fajlove sa jednom ili više anotacija (u zavisnosti od broja alata na slici) iz skupa {elektro busilica, aku busilica, elektro brusilica, aku brusilica}.  


Pozicionirati se u conda env tf15 sa komandom 'activate tf15' i nakon toga postaviti PYTHONPATH i PATH sa komandama 
```
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
```
set PATH=%PATH%;PYTHONPATH
```
U slučaju da se putanje foldera razlikuju od iznad naznačenih, promeniti ih u odgovarajuće.  

Pozicionirati se u models/research folder i uneti sledeću protobuf komandu: 
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
```

Nakon toga jos je potrebno pokrenuti naredne dve komande:  
```
python setup.py build
```

```
python setup.py install
```

Kako bi tensorflow trener mogao da prepozna format koordinata i generisao TFRecords potrebno je pokrenuti 'xml_to_csv.py' skriptu koja će generisati 2 csv fajla u 'images' folderu. Posle toga je potrebno pokrenuti dve skripte za generisanje record fajlova koji služe kao ulaz za tensorflow trener, pozicioniranjem konzole u research/object_detection folder, potrebno je pokrenuti sledeće dve skripte:
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```  
Sada je potrebno promeniti putanje u fajlu 'faster_rcnn_inception_v2_pets.config' koji se nalazi u training folderu. Sledece putanje je potrebno prilagoditi:  
 - Line 123 : input_path: "C:/tensorflow1/models/research/object_detection/train.record"  
 - Line 125 : label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
 - Line 135 : input_path: "C:/tensorflow1/models/research/object_detection/test.record"  
 - Line 137 : label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"  
 - Line 106 : fine_tune_checkpoint: "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"  
   
 Kako bi se pokrenuo sam trening, potrebno je pozicionirati se u models/research/object_detection folder i pokrenuti sledeću komandu:  
 ```
 python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
 ```
 Tensorflow pravi checkpoints svakih desetak minuta, kada loss value spadne na dovoljno malu vrednost moguće je prekinuti trening komandom CTRL+C.  
 Pred sam kraj, potrebno je još eksportovati inference graph ispod navedenom komandom, gde je 'XXXX' zamenjem sa brojem poslednjeg koraka checkpoint-a u folderu 'training' (npr. ako je poslednji checkpoint 'model.ckpt-94192', treba 'XXXX' zameniti sa '94192':  
 ```
 python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
 ```
 Nakon toga, detektor je spreman sa korišćenje.
