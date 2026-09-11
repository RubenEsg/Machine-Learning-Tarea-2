# Etapa 4 — Despliegue en Kubernetes local (Minikube)

Dos manifiestos despliegan la imagen de la etapa 3 en un clúster local:

- **Deployment `heart-model`:** 2 réplicas de la API, con *probes* de disponibilidad y de vida sobre `/health`, límites de recursos y un `securityContext` que impide ejecutar como root.
- **Service `heart-service`:** tipo `LoadBalancer`; expone la API en el puerto 80 y la redirige al puerto 8000 del contenedor, repartiendo el tráfico entre las réplicas.

```{literalinclude} ../k8s/deployment.yaml
:language: yaml
:caption: k8s/deployment.yaml
```

```{literalinclude} ../k8s/service.yaml
:language: yaml
:caption: k8s/service.yaml
```

La imagen es local, así que se carga en Minikube en lugar de descargarla de Docker Hub. Por eso se usa `image: heart-api:latest` con `imagePullPolicy: IfNotPresent`: con la etiqueta `latest`, Kubernetes intentaría descargarla por defecto. Si se publica en Docker Hub, basta con cambiar la imagen por `<TU_USUARIO_DOCKER>/heart-api`, como en el ejemplo del enunciado.

```bash
minikube start --driver=docker
docker build -t heart-api -f docker/Dockerfile .
minikube image load heart-api:latest      # copia la imagen local al clúster
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc
minikube service heart-service --url      # o "minikube tunnel" para la IP externa
```

## Verificación

Se ejecutó en un equipo con Windows 11, Docker 29.7, Minikube 1.39 y Kubernetes 1.37:

```text
$ kubectl rollout status deployment/heart-model
deployment "heart-model" successfully rolled out

$ kubectl get pods -l app=heart-model
NAME                           READY   STATUS    RESTARTS   AGE
heart-model-69dcf79d4d-4b8kj   1/1     Running   0          12s
heart-model-69dcf79d4d-67hdc   1/1     Running   0          12s

$ kubectl get svc heart-service
NAME            TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
heart-service   LoadBalancer   10.106.149.104   <pending>     80:31708/TCP   12s

$ curl -X POST <URL de minikube service>/predict -H "Content-Type: application/json" \
       -d '{"features": [54, "M", "ASY", 140, 239, 0, "Normal", 160, "N", 1.2, "Flat"]}'
{"heart_disease_probability":0.7764260869585767,"prediction":1,"threshold":0.5}
```

- **Disponibilidad:** los 2 pods quedaron `Running` y listos (1/1), lo que confirma que las *probes* sobre `/health` responden y que la imagen corre sin root.
- **IP externa:** aparece como `<pending>` porque en Minikube un `LoadBalancer` solo recibe IP con `minikube tunnel`; `minikube service heart-service --url` abre un acceso local equivalente.
- **Balanceo de carga:** de 20 peticiones enviadas dentro del clúster al puerto 80 del Service, cada réplica atendió aproximadamente la mitad.
- **Autorrecuperación:** al eliminar un pod a propósito, el Deployment creó uno nuevo en 3 segundos, listo a los 12 segundos, y volvió a 2/2 réplicas disponibles.
