import json
import tqdm
import os
import shutil

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, Vt, UsdShade

from .usd_tools import IsEmpty


# -------------------------- JSON file tools --------------------------

def get_json_data(json_path):
    # json_data is a list, each element is a dictionary
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def recovery(annotation_data, duplicates_data):
    recovery_info = []
    for item in annotation_data:
        name = item['name'].replace("_MightBeGlass", "")
        feedback = item['feedback']
        is_dup = False
        for key, value in duplicates_data.items():
            if key == "to_delete":
                continue
            if name in value:
                is_dup = True
                for dup_name in value:
                    recovery_info.append({"name": dup_name, "feedback": feedback})
                break
        if not is_dup:
            recovery_info.append({"name": name, "feedback": feedback})
    # with open("recovery.json", 'w') as f:
    #     json.dump(recovery_info, f, indent=4)
    return recovery_info

def save_json_data(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_json_data_CN(json_path, data):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def convert_list_to_dict(json_file_path):
    with open(json_file_path, 'r') as file:
        data_list = json.load(file)
    result_dict = {}
    for item in data_list:
        for key in item:
            result_dict[key] = item[key]
    return result_dict


# -------------------------- Create usd assests tools --------------------------

transforms_name_set = set([
    "xformOp:transform",
    "xformOp:translate",
    "xformOp:orient",
    "xformOp:scale",
    "xformOpOrder"
])



def check_prim_mdls(prim):
    prim_mdls = []
    prim_name = prim.GetName()
    children = prim_children(prim)
    for child in children:
        relationship = child.GetRelationship("material:binding")
        if relationship:
            target = relationship.GetTargets()
            mdl_name = str(target[0]).split('/')[-1] + '.mdl'
            prim_mdls.append(mdl_name)
            
    return list(set(prim_mdls))


def is_absolute(path):
    if path == '':
        return True
    if path[0] == '/':
        return True
    return False


def prim_children(prim):
    if not prim:
        return []
    prim_list = []
    def collect_prim_children(prim):
        prim_list.append(prim)
        for child in prim.GetChildren():
            collect_prim_children(child)
    collect_prim_children(prim)
    return prim_list


def simplify_path(path):
    # print(path)
    elements = path.split('/')
    simp = []
    for e in elements:
        if e == '':
            # continue
            simp.append(e)
        elif e == '.':
            if len(simp) == 0:
                simp.append(e)
            continue
        elif e == '..':
            if len(simp) != 0 and simp[-1] != '..' and simp[-1] != '.':
                simp = simp[:-1]
            else:
                simp.append(e)
        else:
            simp.append(e)
    new_path = '/'.join(simp)
    return new_path


def copyfile(src, dest):
    srct = simplify_path(src)
    destt = simplify_path(dest)

    try:
        if not os.path.exists(srct):
            raise FileNotFoundError(f"{srct} does not exist! (original: {src})")     
        os.makedirs(os.path.dirname(destt), exist_ok=True)
        shutil.copy2(srct, destt)

    except FileNotFoundError as e:
        print(e)
        return False
    
    return True


def remove_parent_prefix(path):
    elements = path.split('/')
    for i, e in enumerate(elements):
        if e == '..':
            continue
        else:
            elements = elements[i:]
            new_path = os.path.join(*elements)
            return new_path


def recursive_copy(prim_a, prim_b, copy_transform=True, src_prepend='.', dest_prepend='.', ref_prepend='.'):
    '''
        There are three kinds of properties could appear in a prim:
        1. attribute: some attributes
        2. relationship: internal prim path
        3. reference: external usd file path
    '''
    looks_b_path = "/Root/Looks"
    others_b_path = "/Root/Others"
    stage_a = prim_a.GetStage()
    stage_b = prim_b.GetStage()
    # print(prim_a.GetName(), prim_a.GetReferences(), prim_a.HasAuthoredReferences())
    

    ########## process relationships
    rels_a = prim_a.GetRelationships()    
    for rel in rels_a:
        rel_name = rel.GetName()
        targets = rel.GetTargets()
        if len(targets) > 0:
            rel_b = prim_b.CreateRelationship(rel_name, custom=False)
        if rel_name == 'material:binding':
            for t in targets:
                material_a = stage_a.GetPrimAtPath(t)
                material_name = material_a.GetName()
                material_b_path = os.path.join(looks_b_path, material_name)
                material_b = stage_b.GetPrimAtPath(material_b_path)
                # print(Usd.Object.IsValid(p))
                if not Usd.Object.IsValid(material_b):
                    material_b = stage_b.DefinePrim(material_b_path, material_a.GetTypeName())
                    recursive_copy(material_a, material_b, src_prepend=src_prepend, dest_prepend=dest_prepend)
                rel_b.AddTarget(material_b_path)
                strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel)
                UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(rel_b, strength)
                pass
        elif rel_name == 'physics:body0' or rel_name == 'physics:body1':
            for t in targets:
                rel_p_a = stage_a.GetPrimAtPath(t)

                if not Usd.Object.IsValid(rel_p_a):
                    print("invalid target", t)
                    continue
                rel_p_a_name = rel_p_a.GetName()
                rel_p_b_path = os.path.join("/Root", "Instance", rel_p_a_name)
                rel_b.AddTarget(rel_p_b_path)
        else:
            for t in targets:
                rel_p_a = stage_a.GetPrimAtPath(t)
                rel_p_a_name = rel_p_a.GetName()
                rel_p_b_path = os.path.join(others_b_path, rel_p_a_name)
                rel_p_b = stage_b.GetPrimAtPath(rel_p_b_path)
                if not Usd.Object.IsValid(rel_p_b):
                    rel_p_b = stage_b.DefinePrim(rel_p_b_path, rel_p_a.GetTypeName())
                    recursive_copy(rel_p_a, rel_p_b, src_prepend=src_prepend, dest_prepend=dest_prepend)
                
                rel_b.AddTarget(rel_p_b_path)

    ref_prepend = '.'
    prim = prim_a
    while Usd.Object.IsValid(prim):
        if prim.HasAuthoredReferences():
            tmp_references_list = []
            for prim_spec in prim.GetPrimStack():
                tmp_references_list.extend(prim_spec.referenceList.prependedItems)
            # print('tmp_references_list',tmp_references_list, prim_a.GetPath(), prim.GetPath())
            tmp_ref_prepend_path = os.path.join(*(str(tmp_references_list[0].assetPath).split('/')[:-1]))
            ref_prepend = simplify_path(os.path.join(tmp_ref_prepend_path, ref_prepend))
        
        prim = prim.GetParent()

    ########## process attributes
    attributes = prim_a.GetAttributes()
    skip_attrs = set()
    for attr in attributes:
        name = attr.GetName()
        typename = attr.GetTypeName()
        val = attr.Get()
        str_value = str(val)
        input_attr_name = name if name.startswith('inputs:') else f"inputs:{name}"

        if name in skip_attrs:
            continue
        connections = attr.GetConnections()
        
        if not copy_transform and name in transforms_name_set:
            continue
        
        new_attr = prim_b.CreateAttribute(name, typename, custom=False)
        if prim_a.IsA(UsdPhysics.PrismaticJoint) or prim_a.IsA(UsdPhysics.RevoluteJoint):
            if name == 'physics:jointEnabled':
                val = 0

        if val is not None:
            new_attr.Set(val)
            if prim_a.HasAttribute(input_attr_name):
                skip_attrs.add(input_attr_name)
                if prim_b.HasAttribute(input_attr_name):
                    new_input_attr = prim_b.GetAttribute(input_attr_name)
                else:
                    new_input_attr = prim_b.CreateAttribute(input_attr_name, typename, custom=False)
                new_input_attr.Set(val)
                

        if len(connections) > 0:
            for c in connections:
                attr_father_path = str(prim_a.GetPath())
                relative_path = str(c)[len(attr_father_path)+1:]

                dest_father_path = str(prim_b.GetPath())
                connection_b = os.path.join(dest_father_path, relative_path)
                new_attr.AddConnection(connection_b)

        if typename == 'asset':
            colorspace = attr.GetColorSpace()
            if len(colorspace) > 0:
                new_attr.SetColorSpace(colorspace)
            
            if val is None:
                continue
            filepath = val.path
            
                
            if is_absolute(filepath) or str_value.split('@')[1] == "OmniPBR.mdl":
                new_attr.Set(Sdf.AssetPath(filepath))
            else:
            
                # updated_filepath = simplify_path(os.path.join(ref_prepend, filepath))
                updated_filepath = remove_parent_prefix(filepath)
                # updated_filepath = filepath
                # print(prim_a.GetPath(), attr.GetName(), ref_prepend, filepath, updated_filepath, src_prepend)
                new_attr.Set(Sdf.AssetPath(updated_filepath))
                
                # print(prim_a.GetName(), name, src_prepend, filepath)
                source_file_path = os.path.join(src_prepend, ref_prepend, filepath)
                # dest_file_path = os.path.join(dest_prepend, ref_prepend, filepath)
                dest_file_path = os.path.join(dest_prepend, updated_filepath)
                # print(dest_prepend,'=====================', ref_prepend)
                success = copyfile(source_file_path, dest_file_path)



    ########## process geometry primvars
    if prim_a.IsA(UsdGeom.Mesh):
        api_a = UsdGeom.PrimvarsAPI(prim_a)
        api_b = UsdGeom.PrimvarsAPI(prim_b)
        primvars = api_a.GetPrimvars()
        for var in primvars:
            # print(var.GetName(), var.GetInterpolation())
            name = var.GetName()
            it = var.GetInterpolation()
            if it != 'constant':
                var_b = api_b.GetPrimvar(name)
                var_b.SetInterpolation(it)
        # print(primvars)

    ########## process children 
    children = prim_a.GetChildren()
    prim_b_path = str(prim_b.GetPath())
    for child in children:
        typename = child.GetTypeName()
        name = child.GetName()
        # print(name)
        child_path = os.path.join(prim_b_path, name)
        new_child = stage_b.DefinePrim(child_path, typename)
        recursive_copy(child, new_child, src_prepend=src_prepend, dest_prepend=dest_prepend)

    pass


def create_instance(prim, path, src_prepend, dest_prepend):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # print(prim.GetTypeName())
    typename = prim.GetTypeName()
    root = stage.DefinePrim("/Root", "Xform")
    instance = stage.DefinePrim("/Root/Instance", typename)
    looks = stage.DefinePrim("/Root/Looks", "Scope")
    # others = stage.DefinePrim("/Root/Others", "Scope")
    stage.SetDefaultPrim(root)
    recursive_copy(prim, instance, False, src_prepend=src_prepend, dest_prepend=dest_prepend)
    stage.GetRootLayer().Save()


def parse_scene(filepath, dest_path, scene_name_given=None):
    filename = filepath.split('/')[-1]
    src_prepend = "/".join(filepath.split('/')[:-1])
    print(src_prepend)
    stage = Usd.Stage.Open(filepath)

    if scene_name_given is None:
        scene_name = filename.split('.')[0]
    else:
        scene_name = scene_name_given

    prims= []

    prims_instances_parts = stage.GetPrimAtPath('/World/Instances/Part').GetChildren()
    prims_instances_combineds = stage.GetPrimAtPath('/World/Instances/Combined').GetChildren()
    prims_instances_completes = stage.GetPrimAtPath('/World/Instances/Complete').GetChildren()

    prims.extend(prims_instances_parts)
    prims.extend(prims_instances_combineds)
    prims.extend(prims_instances_completes)

    for prim in tqdm.tqdm(prims):

        prim_name = str(prim.GetPath()).split('/')[-1]
        
        dest_prim_path = os.path.join(dest_path, 'models', prim_name)
        if not os.path.exists(dest_prim_path):
            os.makedirs(dest_prim_path)
        if not os.path.exists(dest_prim_path):
            os.makedirs(dest_prim_path)
            os.makedirs(os.path.join(dest_prim_path, "Materials"))
            
        prim_mdls = check_prim_mdls(prim)
        dest_model_path = os.path.join(dest_prim_path, "instance.usd")
        if not os.path.exists(dest_model_path):
            for f in prim_mdls:
                f_name = f.split(".")[0]
                f_src_path = os.path.join(src_prepend, "Materials", f)
                f_dest_path = os.path.join(dest_prim_path, "Materials", f)
                copyfile(f_src_path, f_dest_path)


                # Also copy the same f_name's folder to the dest_path
                f_folder_path = os.path.join(src_prepend, "Materials", f_name)
                f_folder_dest_path = os.path.join(dest_prim_path, "Materials", f_name)
                try:
                    shutil.copytree(f_folder_path, f_folder_dest_path)
                except Exception as e:
                    pass

                # The templates folder must be copy
                templates_folder_path = os.path.join(src_prepend, "Materials", "templates")
                templates_folder_dest_path = os.path.join(dest_prim_path, "Materials", "templates")
                try:
                    shutil.copytree(templates_folder_path, templates_folder_dest_path)
                except Exception as e:
                    pass
                
            create_instance(prim, dest_model_path, src_prepend=src_prepend, dest_prepend=dest_prim_path)
        pass
