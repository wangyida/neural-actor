##-*****************************************************************************
##
## Copyright (c) 2009-2016,
##  Sony Pictures Imageworks Inc. and
##  Industrial Light & Magic, a division of Lucasfilm Entertainment Company Ltd.
##
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
## *       Redistributions of source code must retain the above copyright
## notice, this list of conditions and the following disclaimer.
## *       Redistributions in binary form must reproduce the above
## copyright notice, this list of conditions and the following disclaimer
## in the documentation and/or other materials provided with the
## distribution.
## *       Neither the name of Industrial Light & Magic nor the names of
## its contributors may be used to endorse or promote products derived
## from this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
##-*****************************************************************************


FIND_PACKAGE(Imath)

IF (Imath_FOUND)
    MESSAGE(STATUS "Found package Imath")
    SET(ALEMBIC_ILMBASE_LIBS Imath::Imath)
    SET(ALEMBIC_USING_IMATH_3 ON)
ELSE()
    MESSAGE(STATUS "Could not find Imath looking for IlmBase instead.")
    # What we really want to do is look for libs Imath and half
    FIND_PACKAGE(IlmBase)
    SET(ALEMBIC_USING_IMATH_3 OFF)

    IF (ILMBASE_FOUND)
        SET(ALEMBIC_ILMBASE_FOUND 1 CACHE STRING "Set to 1 if IlmBase is found, 0 otherwise")

        SET(ALEMBIC_ILMBASE_LIBS
            ${ALEMBIC_ILMBASE_IMATH_LIB}
            ${ALEMBIC_ILMBASE_ILMTHREAD_LIB}
            ${ALEMBIC_ILMBASE_IEX_LIB}
            ${ALEMBIC_ILMBASE_HALF_LIB}
        )

        if (${ALEMBIC_ILMBASE_IEXMATH_LIB})
            SET(ALEMBIC_ILMBASE_LIBS ${ALEMBIC_ILMBASE_LIBS} ${ALEMBIC_ILMBASE_IEXMATH_LIB})
        endif (${ALEMBIC_ILMBASE_IEXMATH_LIB})

    ELSE()
        SET(ALEMBIC_ILMBASE_FOUND 0 CACHE STRING "Set to 1 if IlmBase is found, 0 otherwise")
    ENDIF()
ENDIF()
