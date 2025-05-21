# Annotation Tools

## Instance Annotation Tool
This annotation tool is mainly used to classify structures or instances

**Data Directory Structure**

The directory structure should be as below.

    data
    ├── part1
    │   ├── index_usd
    │   │   ├── 04d
    │   │   │   ├── pcs
    │   │   │   │   ├── duplicate_records
    │   │   │   │   │    ├── duplicate_records.json
    │   │   │   │   ├── mesh_scene
    │   │   │   │   │    ├── scene_name.ply
    │   │   │   │   ├── object.npy
    │   │   │   │   ├── ...
    │   │   │   ├── annotation.json
    │   │   ├── ...
    │   ├── ...
    ├── part2
    │   ├── ...

**Viser Server point size**
- Server1: 
    - The whole scene point cloud, point_size = 0.0001
    - The bbox line point cloud, point_size = 0.00001
    - The highlighted point cloud, point_size = 0.0005
- Server2: 
    - The single object point cloud, point_size = 0.003


## Region Annotation Tool
This annotation tool is mainly used to classify regions.

**Data Directory Structure**

The directory structure should be as below.

    data
    ├── part1
    │   ├── index_usd
    │   │   ├── 04d
    │   │   │   ├── bevmap_0
    │   │   │   │   ├── bev_map.jpg
    │   │   │   │   ├── pos.json
    │   │   │   ├── regions_0
    │   │   │   │   ├── sample_0
    │   │   │   │   │   ├── combine_1.jpg
    │   │   │   │   │   ├── combine_2.jpg
    │   │   │   │   ├──sample_1
    │   │   │   │   │   ├── combine_1.jpg
    │   │   │   │   │   ├── combine_2.jpg
    │   │   │   │   ├──...
    │   │   ├── ...
    │   ├── ...
    ├── part2
    │   ├── ...