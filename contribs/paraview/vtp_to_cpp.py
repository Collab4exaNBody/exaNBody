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
fout.write( "// file generated from VTK PolyData file %s using vtp_to_cpp.py\nstatic constexpr unsigned long n_vertices=%d;\ndouble vertices[n_vertices][3] = {\n" % (inputfile,len(pts)) )
sep = "  "
for v in pts:
  fout.write( "%s{ % .9e , % .9e , % .9e }\n" % (sep,v[0],v[1],v[2]) )
  sep = ", "

fout.write( "};\n" )

nd=1
m=10
while len(polys)>m:
  nd = nd + 1
  m = m * 10
trifmt = "%%s{ %%%dd , %%%dd , %%%dd }\n" % (nd,nd,nd)
sep = "  "

nt = len(polys) // 4
fout.write( "static constexpr unsigned long n_triangles=%d;\nunsigned long triangles[n_triangles][3] = {\n" % nt)
for t in range(nt):
  np = polys[t*4+0]
  assert(np==3)
  p1 = polys[t*4+1]
  p2 = polys[t*4+2]
  p3 = polys[t*4+3]
  fout.write( trifmt % (sep,p1,p2,p3) )
  sep = ", "

fout.write( "};\n" )
fout.close()

