#!/usr/bin/env python3
# Submitted as a Dataproc PySpark job, so spark-submit runs it with PYSPARK_PYTHON
# and this shebang is never read. It used to name the 2.x images' conda Python,
# which the 3.0 images (Pixi) do not have.

import sys

from pts.core import main

if __name__ == '__main__':
    if sys.argv[0].endswith('-script.pyw'):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
