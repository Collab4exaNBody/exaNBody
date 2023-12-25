#!/bin/bash

sed 's/[ \t]*#[ \t]*pragma[ \t]\+\([^\n]*\)/_Pragma(\"\1\");/g' | clang-format | sed -e 's/ \([ \t]*\)_Pragma(\"\(.*\)\");/#\1pragma \2/g' -e 's/_Pragma(\"\(.*\)\");/#pragma \1/g'


AlignConsecutiveShortCaseStatements

