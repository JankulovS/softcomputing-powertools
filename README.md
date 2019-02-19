# softcomputing-powertools


Korišćena Ananconda 3.7 (python 3.6.8) i kreirano virtualno okruženje tf15 sa komandom 'conda create -n tf15 pip'

Potrebne biblioteke:

tensorflow-gpu
pandas
opencv-python


Testirano na Windows 10
Trenirano na Ryzen 5 1500X
nVidia GTX 960 4GB

# Pokretanje već treniranog detektora

Skinuti tensorflow models github repozitorijum sa https://github.com/tensorflow/models i ekstraktovati ga u željeni folder, nakon toga skinuti ovaj repozitorijum i ekstraktovati ga u models/research/object_detection. Skinuti folder 'training' sa https://drive.google.com/drive/folders/1GS79qu6LL9NgS32D7HtUZ-FvdqGwj4Ny i postaviti ga u models/research/object_detection, takođe skinuti folder 'inference_graph' i njega postaviti u models/research/object_detection. Pokrenuti konzolu i pozicionirati se u conda env tf15 sa komandom 'activate tf15' i nakon toga postaviti PYTHONPATH sa komandom 
```
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
```
U slučaju da se putanje foldera razlikuju od iznad naznačenih, promeniti ih u odgovarajuće.
Uneti komandu 'idle' u konzolu kako bi se otvorio novi python shell, otvoriti skriptu object_detection_image.py i pokrenuti je. Ako je sve u redu, nakon desetak sekundi otvoriće se novi prozor sa slikom i označenim alatom. Za drugi unos potrebno je samo promeniti konstantu IMAGE_NAME u odgovarajuću sliku koja se nalazi u folderu skripte, takođe radi na .png formatu.

# Treniranje

