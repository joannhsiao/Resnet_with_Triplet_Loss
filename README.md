# Resnet_with_Triplet_Loss
## Info
- Insight face dataset: Asian-Celeb (93978 Identities, 2830146 Images)
- DataLog: sorted data
- Model: Resnet v50
- Loss function: Triplet Loss

## Dataset Distribution
  ![圖片4](https://user-images.githubusercontent.com/59599987/177514291-c5703e57-998d-4342-a1e8-4a8892686887.png)

## Network Arichecture (Resner + Triplet Loss)
  ![螢幕擷取畫面 2022-07-06 170453](https://user-images.githubusercontent.com/59599987/177514310-eed8950b-f377-42fb-ab13-9a02b30e36df.png)

## Result
1. Cosine similarity:

  ![圖片1](https://user-images.githubusercontent.com/59599987/177514478-afe05b42-8fbf-4b3f-a908-c04a3496948a.jpg)

2. Distributions of Intra- and Inter-class distances:

  ![圖片2](https://user-images.githubusercontent.com/59599987/177514505-eb0bc62d-ee14-4432-bb0d-8c26615f7802.jpg)

3. Accuracy over time:

  ![圖片3](https://user-images.githubusercontent.com/59599987/177514519-d9d94f20-4b73-4649-8b6a-26cb8e75efb0.png)
