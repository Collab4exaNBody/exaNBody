#!/bin/bash

SCRIPT_PATH=`dirname $0`
CFCONF=`realpath ${SCRIPT_PATH}/..`/.clang-format

if [ -f "$1" ]
then
  STDIN="$1"
  STDOUT="$1.code-format"
  MVCMD="mv -f ${STDOUT} ${STDIN}"
else
  STDIN="/dev/stdin"
  STDOUT="/dev/stdout"
  MVCMD="test"
fi

sed -e 's/[ \t]*#[ \t]*pragma[ \t]\+\(.*\)/_Pragma(\"\1\");/g'          \
    -e 's/[ \t]*#[ \t]*ifdef[ \t]*\(.*\)/PPIF_BEGIN(\"ifdef \1\")/g'    \
    -e 's/[ \t]*#[ \t]*ifndef[ \t]*\(.*\)/PPIF_BEGIN(\"ifndef \1\")/g'  \
    -e 's/[ \t]*#[ \t]*if[ \t]*\(.*\)/PPIF_BEGIN(\"if \1\")/g'          \
    -e 's/[ \t]*#[ \t]*else/PPIF_END(\"else\")\nPPIF_BEGIN(\"else\")/g' \
    -e 's/[ \t]*#[ \t]*endif/PPIF_END()/g'                              \
    -e 's/[ \t]*#[ \t]*include[ \t]*\"\(.*\)\"/PPINCLUDE(\"\1\")/g'    \
    -e 's/[ \t]*#[ \t]*include[ \t]*<\(.*\)>/PPINCLUDE(\"<\1>\")/g'    \
    -e 's/[ \t]*#[ \t]*define[ \t]*\(.*\)/PPDEFINE(R\"EOF(\1)EOF\");/g' \
< ${STDIN}                                                              \
|                                                                       \
clang-format --assume-filename=${CFCONF}                                \
|                                                                       \
sed -e 's/ \([ \t]*\)_Pragma(\"\(.*\)\");/#\1pragma \2/g'               \
    -e 's/_Pragma(\"\(.*\)\");/#pragma \1/g'                            \
    -e 's/ \([ \t]*\)PPIF_BEGIN(\"\(.*\)\")/#\1\2/g'                    \
    -e 's/PPIF_BEGIN(\"\(.*\)\")/#\1/g'                                 \
    -e 's/ \([ \t]*\)PPIF_END()/#\1endif/g'                             \
    -e 's/PPIF_END()/#endif/g'                                          \
    -e '/PPIF_END(\"else\")/d'                                          \
    -e 's/ \([ \t]*\)PPINCLUDE(\"<\(.*\)>\")/#\1include <\2>/g'         \
    -e 's/PPINCLUDE(\"<\(.*\)>\")/#include <\1>/g'                      \
    -e 's/ \([ \t]*\)PPINCLUDE(\"\(.*\)\")/#\1include \"\2\"/g'         \
    -e 's/PPINCLUDE(\"\(.*\)\")/#include <\1>/g'                        \
    -e 's/ \([ \t]*\)PPDEFINE(R\"EOF(\(.*\))EOF\")/#\1define \2/g'      \
    -e 's/PPDEFINE(R\"EOF(\(.*\))EOF\")/#define \1/g'                   \
> ${STDOUT} && ${MVCMD}

