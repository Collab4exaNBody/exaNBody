#!/usr/local/ParaView/bin/pvpython

import sys
import vtk
from vtk.util.numpy_support import vtk_to_numpy

inputfile = sys.argv[1]
outputfile = sys.argv[2]
print("Export %s to %s ..." % (inputfile,outputfile) )

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName( inputfile )
reader.Update()

triangulate = vtk.vtkTriangleFilter()
triangulate.SetInputData( reader.GetOutput() )
triangulate.Update()
pdo = triangulate.GetOutput()

pts = vtk_to_numpy( pdo.GetPoints().GetData() )
polys = vtk_to_numpy( pdo.GetPolys().GetData() )

fout = open(outputfile,"w")
fout.write( "# file generated from VTK PolyData file %s using vtp_to_trimesh.py tool\ntrimesh_init:\n  mesh:\n    vertices:\n" % (inputfile) )
for v in pts:
  fout.write( "      - [ % .9e , % .9e , % .9e ]\n" % (v[0],v[1],v[2]) )

nd=1
m=10
while len(polys)>m:
  nd = nd + 1
  m = m * 10
trifmt = "      - [ %%%dd , %%%dd , %%%dd ]\n" % (nd,nd,nd)

fout.write( "    triangles:\n" )
nt = len(polys) // 4
for t in range(nt):
  np = polys[t*4+0]
  assert(np==3)
  p1 = polys[t*4+1]
  p2 = polys[t*4+2]
  p3 = polys[t*4+3]
  fout.write( trifmt % (p1,p2,p3) )

fout.close()

