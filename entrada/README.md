# entrada/

Deja aquí las fotos (o archivos con templates) que quieras probar contra el sistema
desplegado en Docker. La carpeta se monta en el contenedor, en solo lectura, como `/entrada`:

```bash
docker compose exec facerec python -m sistema identificar /entrada/foto.jpg
docker compose exec facerec python -m sistema verificar adele /entrada/foto.jpg
```

Su contenido no se versiona (salvo este archivo).
