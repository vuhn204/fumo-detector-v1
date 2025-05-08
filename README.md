# Detector de Fumos v1.0
Entrena y ejecuta un detector de fumos con YOLOv8.
## Estructura del proyecto
fumo-detector-v1/
├── data.zip # Zip exportado de Label Studio (images/ + labels/)
├── notebooks/
│ └── Entrenamiento_YOLOv8.ipynb # Notebook con todo el flujo de entrenamiento
├── scripts/
│ └── detect_fumos.py # Script Python para detección en vivo
├── models/
│ └── best_fumo_v1.pt # Peso final (se genera tras entrenar)
├── classes.txt # Lista de clases (una por línea): "fumo"
├── requirements.txt # Dependencias necesarias
└── .gitignore # Archivos/carpetas a ignorar


- **data.zip**  
  Archivo ZIP exportado de Label Studio (contiene `images/` y `labels/` en formato YOLO).

- **notebooks/Entrenamiento_YOLOv8.ipynb**  
  Notebook con todo el flujo de entrenamiento en Colab o Jupyter.

- **scripts/detect_fumos.py**  
  Script Python para ejecutar detección en vivo con webcam.

- **models/best_fumo_v1.pt**  
  Peso del modelo resultante (se genera tras entrenar).

- **classes.txt**  
  Lista de clases (una por línea):
  Fumo
- **fumo_config.yaml**  
  Configuración de rutas y clases para YOLOv8. Se genera automáticamente desde el notebook.

- **requirements.txt**  
  Dependencias:
  ultralytics
  opencv-python
  pyyaml
  tqdm

	## Instalación

	1. **Clona este repositorio**  
	```bash
   	git clone https://github.com/vuhn204/fumo-detector-v1.git
  	cd fumo-detector-v1
	2. Instala las dependencias
	pip install -r requirements.txt
	## Entrenamiento
	1. Descomprime tu dataset
	unzip data.zip -d raw_dataset
	2. Abre y ejecuta el notebook
	En Google Colab(recomendado): sube data.zip al entorno y abre notebooks/Entrenamiento_YOLOv8.ipynb.
	En Jupyter local: sitúate en la carpeta raíz y abre el mismo notebook.

	3. Flujo del notebook
	Instala ultralytics y tqdm.
	Descomprime data.zip en raw_dataset/.
	Divide el dataset en:
	 fumo_dataset/
	 ├── train/images
	 ├── train/labels
	 ├── val/images
	 └── val/labels
	Genera el archivo fumo_config.yaml con esta estructura:
	path: fumo_dataset
	train: train/images
	val:   val/images
	nc:    1
	names: ['Fumo']
	Lanza el entrenamiento:
	from ultralytics import YOLO
	model = YOLO('yolov8s.pt')
	model.train(data='fumo_config.yaml', epochs=50, imgsz=640, batch=12,
        	    project='models', name='colab_fumo_v1', exist_ok=True)
	El mejor peso se guarda en:
		models/colab_fumo_v1/weights/best.pt
	Descárgalo y cópialo a models/best_fumo_v1.pt.


	## Inferencia en vivo
	Con tu modelo final en models/best_fumo_v1.pt, ejecuta:
	python scripts/detect_fumos.py
	Este script:
	from ultralytics import YOLO
	import cv2

	model = YOLO('models/best_fumo_v1.pt')
	cap   = cv2.VideoCapture(0)

	while True:
	    ret, frame = cap.read()
	    if not ret: break
	    model.predict(source=frame, show=True, conf=0.4)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	cap.release()
	cv2.destroyAllWindows()

















