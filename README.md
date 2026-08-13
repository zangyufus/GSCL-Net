# GSCL-Net: Geometry-Spectral Collaborative Learning with Direction-Aware Context Construction for Airborne Multispectral Point Cloud Classification
This paper introduces GSCL-Net, a geometry-spectral collaborative learning network with direction-aware context construction for fine-grained land-cover classification of airborne multispectral point clouds. Airborne multispectral point clouds often present structural complexity and spectral ambiguity, while existing methods still struggle to fully exploit geometry-spectral complementarity due to inadequate local context modeling and cross-modal coordination. To address these limitations, GSCL-Net integrates three core components: a Height-Guided Adaptive Edge Convolution (HGA-EdgeConv), a Direction-Aware Spectral Neighborhood Selection (DSNS) module, and a Bilateral Feature-Offset Fusion (BOF) module. The network is specifically designed to handle complex airborne scenes by adaptively aggregating geometric features based on local vertical configurations, constructing highly representative spectral contexts, and sequentially calibrating multimodal features to reduce cross-branch conflicts. Extensive experiments demonstrate that GSCL-Net effectively enhances discriminative feature learning by preserving and leveraging complementary information. Overall, GSCL-Net provides a robust collaborative learning solution for accurate land-cover classification and supports advanced airborne multispectral LiDAR point cloud analysis.
## 🗂️ Dataset
Experiments were conducted on two airborne multispectral point cloud datasets: the NUIST dataset and the Whitchurch–Stouffville (WS) dataset. The NUIST dataset was acquired over the campus of Nanjing University of Information Science and Technology (NUIST), Nanjing, China, and the pointwise ground-truth labels were manually annotated by our team. 
It represents a dense urban campus containing buildings, roads, vegetation, and open ground, resulting in complex object boundaries and mixed neighborhoods. By contrast, the WS dataset is an external multispectral LiDAR dataset acquired over Whitchurch–Stouffville, Ontario, Canada, and has been used in previous studies for benchmarking multispectral LiDAR land-cover classification methods. 
For the NUIST dataset, nine sample areas (Areas 1–2, 4–7, and 9–11) were used for training, while Areas 3 and 8 were reserved for testing. For the WS benchmark dataset, Areas 1–10 were used for training and Areas 11–12 were used for testing. To support batch training on the large-scale scenes, both datasets were partitioned into 20m×20m spatial blocks. Each block was then sampled to 2,048 points, with random downsampling for dense blocks and repeated sampling for sparse blocks.
<p align="center">
<img width="906" height="392" alt="image" src="https://github.com/user-attachments/assets/40b84487-29a6-41a0-933f-d593795f149e" />
<img width="906" height="343" alt="image" src="https://github.com/user-attachments/assets/3286e9c6-ea05-40fd-88ab-f12f7190eebc" />
<img width="1144" height="386" alt="image" src="https://github.com/user-attachments/assets/5abc4e31-7bb9-41c9-98d6-2d8e8c14eb97" />
</p>

## 🚀 Method

We propose GSCL-Net for airborne multispectral point cloud classification. The network separately learns geometric and spectral features and leverages complementary relationship to produce discriminative joint representation.  Specifically, the framework is organized into an HGE branch, a DSE branch, and a geometry-spectral fusion branch, whose key modules are HGA-EdgeConv, DSNS, and BOF, respectively. HGA-EdgeConv regulates center-neighbor contributions using pairwise height relationships, DSNS determines neighborhood membership according to the dominant local spectral-spatial variation, and BOF sequentially calibrates cross-branch responses through bilateral feature offsets. Together, these branches form a progressive learning pipeline, producing discriminative pointwise representation.
<p align="center">
<img width="1118" height="371" alt="image" src="https://github.com/user-attachments/assets/1d9db316-72c1-4d51-a7e1-420458c4a5c0" />
</p>

## 💥 Comparison with Existing Methods
Extensive experiments were conducted to evaluate the proposed network against several state-of-the-art methods on both the NUIST and Whitchurch-Stouffville (WS) datasets.
<p align="center">
<img width="1180" height="368" alt="image" src="https://github.com/user-attachments/assets/4af7f327-0ec6-4b84-b63b-4d342791e08f" />
<img width="1125" height="352" alt="image" src="https://github.com/user-attachments/assets/728c6bfe-357d-4508-b864-dc416ca22a03" />
<img width="974" height="761" alt="image" src="https://github.com/user-attachments/assets/4c440ecf-16f5-48ac-9709-7de47b693c23" />
</p>
Visual comparisons on both datasets further demonstrate the superiority of the proposed GSCL-Net. In the NUIST dataset, which features dense urban environments, our method produces highly spatially coherent classification maps. Compared to existing advanced methods, it not only effectively reduces interference in homogeneous regions but also preserves sharper boundaries for buildings and roads. For the WS dataset, the network exhibits exceptional capability in maintaining fine-grained structural details. It successfully preserves slender objects like powerlines and generates predictions that are visually closest to the Ground Truth. Overall, the qualitative results confirm that GSCL-Net effectively mitigates spectral ambiguity and handles complex structural variations in diverse airborne scenes.
<p align="center">
<img width="773" height="797" alt="image" src="https://github.com/user-attachments/assets/278e9085-c3b7-4a30-8099-77e92928017f" />
<img width="699" height="803" alt="image" src="https://github.com/user-attachments/assets/eee92822-b744-4e69-8a7a-ad589ad76918" />
</p>
Coming in future updates.
