import zipfile
import os

zip_path = 'report/CVPR2022_Shadow_Remover_Writeup.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk('report'):
        for file in files:
            if file.endswith('.zip'):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, 'report')
            z.write(full_path, rel_path)

print('Zip created cleanly, size:', os.path.getsize(zip_path))

