#!/usr/bin/env python
import os
import vtk
import argparse
import trimesh
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy

def sample_pc_from_mesh(file_path, n_points, sample_normals = True):
    mesh = trimesh.load(file_path)

    mesh.apply_translation(-mesh.center_mass)
    scaling_factor = pow(3 / 4 * mesh.bounding_sphere.volume, 1 / 3)
    mesh.apply_scale(1 / scaling_factor)

    #try:
        #samples, normals = trimesh.sample.sample_surface(mesh, n_points, seed=42)
    #except:
    #     if not mesh.is_watertight:
    #         verts, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, 50000)
    #         mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    #     samples, normals = trimesh.sample.sample_surface(mesh, n_points, seed=42)

    samples, normals = trimesh.sample.sample_surface(mesh, n_points, seed=42)
    samples = np.array(samples)
    if sample_normals:
        normals = np.array(mesh.face_normals[normals])
        samples = np.concatenate([samples, normals], axis=1)
    return samples, scaling_factor


def convertFile(filepath):
    #if not os.path.isdir(outdir):
    #    os.makedirs(outdir)
    if os.path.isfile(filepath):
        basename = os.path.basename(filepath)
        print("Copying file:", basename)
        basename = os.path.splitext(basename)[0]
        outfile = "tmp2.stl" #os.path.join(outdir, basename+".stl")
        reader = vtk.vtkGenericDataObjectReader()
        #vtk.vtkGenericDataObjectReader()
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllTensorsOn()
        reader.SetFileName(filepath)
        reader.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(reader.GetOutputPort())
        writer.SetFileName(outfile)
        return writer.Write()==1, reader.GetOutput()
    return False

import os.path
def convertFiles(indir):
    files = os.listdir(indir)
    files = [ os.path.join(indir,f) for f in files if f.endswith('.vtk') ]
    ret = 0
    print("In:", indir)
    print("Out:")
    for f in files:
        basename = os.path.basename(f)
        basename = os.path.splitext(basename)[0]
        if os.path.isfile(indir + "/txt8/"+ basename + "_pc.txt"):
            continue
        cnt, seq0 = convertFile(f)
        ret += cnt
        pc, scaling_factor = sample_pc_from_mesh("tmp2.stl", 1024*8)
        seq = vtk_to_numpy(seq0.GetPoints().GetData())/scaling_factor
        min_val, max_val = seq0.GetPointData().GetArray("Potential").GetRange()
        min_val2, max_val2 = seq0.GetPointData().GetArray("NormalPotential").GetRange()
        i = 0
        new_pc = []
        for p in pc:
            i = i + 1
            idx = np.argmin(np.linalg.norm(p[:3]-seq, axis = 1))
            potential = seq0.GetPointData().GetArray("Potential").GetValue(idx)
            potential_normal = seq0.GetPointData().GetArray("NormalPotential").GetValue(idx)
            p = np.append(p, (potential-min_val)/(max_val-min_val))
            p = np.append(p, (potential_normal-min_val2)/(max_val2-min_val2))
            new_pc.append(p)

        np.savetxt(indir + "/txt8/"+ basename + "_pc.txt", np.array(new_pc),  delimiter=",")

    print("Successfully converted %d out of %d files." % (ret, len(files)))


import time

if __name__ == '__main__':
    begin = time.time()
    indir = ".../shrec2025/new_data/train_set_vtk/"
    #indir = ".../shrec2025/new_data/test_set_vtk/"

    convertFiles(indir)
    end = time.time()
    print(end - begin)