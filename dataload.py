import numpy as np
from PIL import Image
import os
from tqdm import tqdm



G_OFFSETS = [(-1,-1), (-1, 0), (-1,1),
             (0, -1), (0, 0), (0, 1),
             (1, -1), (1, 0), (1, 1)
        ]
R_OFFSETS_RG = [(0,0), (-1,-1), (-1,1), (1,-1), (1, 1)]
B_OFFSETS_RG = [(-1,0), (0,1), (1,0), (0,-1)]
B_OFFSETS_BG = [(0,0), (-1,-1), (-1,1), (1,-1), (1, 1)]
R_OFFSETS_BG = [(-1,0), (0,1), (1,0), (0,-1)]



def rearrange_spr(rgb):
    # print(rgb)
    H, W, _ = rgb.shape
    tW = int(W*1.5)
    spr = np.zeros((H, tW, 3))
    rgb = np.asarray(rgb, dtype=int)
    tempdata = np.zeros((H, int(3*W)), dtype=int)

    for i in range(H):
        for j in range(W):
            tempdata[i, 3*j] = rgb[i, j, 0]
            tempdata[i, 3*j+1] = rgb[i, j, 1]
            tempdata[i, 3*j+2] = rgb[i, j, 2]

    for i in range(H):
        for j in range(tW):
            spr[i,j,1] = tempdata[i, 2*j+1]
            if (i^j) & 1 == 1:
                spr[i,j,2] = tempdata[i, 2*j]
            else:
                spr[i,j,0] = tempdata[i, 2*j]

    spr_arr = np.zeros((H+2, tW+2, 3))
    spr_arr[1:H+1, 1:tW+1, :] = spr[:,:,]

    spr_arr[0, 0, :] = spr[0, 0, :]
    spr_arr[0, 1+tW, :] = spr[0, tW-1, :]
    spr_arr[1+tW, 0, :] = spr[tW-1, 0,  :]
    spr_arr[1+tW, 1+tW, :] = spr[tW-1, tW-1, :]

    spr_arr[0, 1:tW, :] = spr[1, 0, :]
    spr_arr[1+tW, 1:tW+1, :] = spr[tW-2, :, :]
    spr_arr[1:1+tW, 0, :] = spr[0:tW, 1, :]
    spr_arr[1:tW+1, 1+tW, :] = spr[0:tW, tW-2, :]
    return spr_arr

def extract_feature(spr_array, rgb_array):
    H, W, _ = rgb_array.shape
    data_rg = []
    data_bg = []
    for i in range(1, H+1):
          for j in range(1, W+1):
              if (i+j) % 2 == 0:
                  f = []
                  for dy, dx in G_OFFSETS:
                      f.append(spr_array[i+dy, j+dx, 1])

                  for dy, dx in R_OFFSETS_RG:
                      f.append(spr_array[i+dy, j+dx, 0])

                  for dy, dx in B_OFFSETS_RG:
                      f.append(spr_array[i+dy, j+dx, 2])

                  label = rgb_array[i-1, j-1, :].tolist()
                  if len(f) == 18 and sum(f) > 0:
                      data_rg.append(f+label)
              else:
                  g = []
                  for dy, dx in G_OFFSETS:
                      g.append(spr_array[i+dy, j+dx, 1])

                  for dy, dx in B_OFFSETS_BG:
                      g.append(spr_array[i+dy, j+dx, 2])

                  for dy, dx in R_OFFSETS_BG:
                      g.append(spr_array[i+dy, j+dx, 0])

                  label = rgb_array[i-1, j-1, :].tolist()
                  if len(g) == 18 and sum(g) > 0:
                      data_bg.append(g+label)

      return np.array(data_rg, dtype=np.float32), np.array(data_bg, dtype=np.float32)

  def process_and_save(input_dir, output_dir, spr_suffix='.bmp', rgb_suffix='.bmp'):
      for fname in os.listdir(os.path.join(input_dir, 'sprPics')):
          if not fname.endswith(spr_suffix):
              continue
          base = fname[:-len(spr_suffix)]
          spr_path = os.path.join(input_dir, 'sprPics', base + spr_suffix)
          rgb_path = os.path.join(input_dir, 'originalPics', base + rgb_suffix)

          # loading data
          spr_array = np.array(Image.open(spr_path))
          rgb_image = np.array(Image.open(rgb_path))

          # extrac feature and label
          spr_array = rearrange_spr(spr_array)
          data_rg, data_bg = extract_feature(spr_array, rgb_image)

          np.save(os.path.join(output_dir, 'rg', f"{base}_RG.npy"), data_rg)
          np.save(os.path.join(output_dir, 'bg', f"{base}_BG.npy"), data_bg)
          print(f"Saved {base}_RG.npy ({data_rg.shape}), Saved {base}_BG.npy ({data_bg.shape})")

  if __name__ == '__main__':
      process_and_save('./Pics', './dataset/')
