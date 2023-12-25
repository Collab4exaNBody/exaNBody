#!/bin/bash

sed -e 's/[ \t]*#[ \t]*pragma[ \t]\+\(.*\)/_Pragma(\"\1\");/g'   \
    -e 's/[ \t]*#[ \t]*ifdef[ \t]*\(.*\)/PPIF_BEGIN(\"ifdef \1\")/g'   \
    -e 's/[ \t]*#[ \t]*ifndef[ \t]*\(.*\)/PPIF_BEGIN(\"ifndef \1\")/g' \
    -e 's/[ \t]*#[ \t]*if[ \t]*\(.*\)/PPIF_BEGIN(\"if \1\")/g'         \
    -e 's/[ \t]*#[ \t]*else/PPIF_END(\"else\")\nPPIF_BEGIN(\"else\")/g' \
    -e 's/[ \t]*#[ \t]*endif/PPIF_END()/g'                \
    -e 's/[ \t]*#[ \t]*include[ \t]*[<\"]\(.*\)[>\"]/PPINCLUDE(\"\1\");/g'                \
    -e 's/[ \t]*#[ \t]*define[ \t]*\(.*\)/PPDEFINE(R"EOF(\1)EOF");/g'                \
    | \
clang-format \
    | \
sed \
    -e 's/ \([ \t]*\)_Pragma(\"\(.*\)\");/#\1pragma \2/g' \
    -e 's/_Pragma(\"\(.*\)\");/#pragma \1/g'
#    -e 's/ \([ \t]*\)PPIF_BEGIN(\(.*\))/#\1\2/g'            \
#    -e 's/PPIF_BEGIN(\(.*\))/#if\1/g'                         \
#    -e 's/ \([ \t]*\)PPIF_END()/#\1endif/g'              \
#    -e 's/PPIF_END()/#endif/g'                           \
#   
