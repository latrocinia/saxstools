#define SQUARE(a) ((a) * (a))


__kernel
void rotate_points(float16 rotmat, __global float4 *xyz, __global float4 *rot_xyz, uint npoints)
{
    uint id = get_global_id(0);
    uint stride = get_global_size(0);

    for (uint i = id; i < npoints; i += stride) {

        rot_xyz[i].s0 = rotmat.s0 * xyz[i].s0 + rotmat.s1 * xyz[i].s1 + rotmat.s2 * xyz[i].s2;
        rot_xyz[i].s1 = rotmat.s3 * xyz[i].s0 + rotmat.s4 * xyz[i].s1 + rotmat.s5 * xyz[i].s2;
        rot_xyz[i].s2 = rotmat.s6 * xyz[i].s0 + rotmat.s7 * xyz[i].s1 + rotmat.s8 * xyz[i].s2;
    }
}


__kernel
void calc_chi2(__global int *interspace,
               __global float *q,
               __global float *Iq,
               __local float *tmpIq,
               __global int *rind,
               __global float4 *rxyz,
               __global int *lind,
               __global float4 *lxyz,
               float4 origin,
               float voxelspacing,
               __global float *fifj,
               __global float *targetIq,
               __global float *sq,
               __global float *chi2,
               uint4 shape,
               uint nq, uint nind1, uint nind2, uint4 fifj_shape
               )
{
    uint id = get_global_id(0);
    uint stride = get_global_size(0);
    uint lid = get_local_id(0);
    uint lws = get_local_size(0);

    uint slice = shape.s2 * shape.s1;
    uint fifj_slice = fifj_shape.s2 * fifj_shape.s1;

    for (uint i = id; i < shape.s3; i += stride) {

        if (interspace[i] == 0)
            continue;

        // set local memory to zero
        for (uint q1 = 0; q1 < nq; q1++)
            tmpIq[q1 * lws + lid] = Iq[q1];

        uint z = i / slice;
        uint y = (i - z*slice)/shape.s2;
        uint x = i - z*slice - y*shape.s2;

        // Calculate the scattering curve
        for (uint n2 = 0; n2 < nind2; n2++) {
            for (uint n1 = 0; n1 < nind1; n1++) {

                float lx = lxyz[n2].s0 + x*voxelspacing + origin.s0;
                float ly = lxyz[n2].s1 + y*voxelspacing + origin.s1;
                float lz = lxyz[n2].s2 + z*voxelspacing + origin.s2;

                float dx = rxyz[n1].s0 - lx;
                float dy = rxyz[n1].s1 - ly;
                float dz = rxyz[n1].s2 - lz;
                float rij = sqrt(SQUARE(dx) + SQUARE(dy) + SQUARE(dz));

                uint fifj_ind = fifj_slice * rind[n1] + fifj_shape.s2 * lind[n2];
                for (uint q1 = 0; q1 < nq; q1++) {

                    float qrij = q[q1] * rij;
                    tmpIq[q1 * lws + lid] += 2 * fifj[fifj_ind + q1] * native_sin(qrij) / qrij;
                }
            }
        }

        // Calculate the scaling factor
        float sum_Iq_targetIq = 0;
        float sum_Iq2 = 0;
        for (uint q1 = 0; q1 < nq; q1++) {
            float sq2 = SQUARE(sq[q1]);
            sum_Iq_targetIq += (tmpIq[q1 * lws + lid] * targetIq[q1]) / sq2;
            sum_Iq2 += SQUARE(tmpIq[q1 * lws + lid]) / sq2;
        }
        float fscale = sum_Iq_targetIq / sum_Iq2;

        // Calculate chi2
        chi2[i] = 0;
        for (uint q1 = 0; q1 < nq; q1++) {
            //chi2[i] += SQUARE((targetIq[q1] - fscale * tmpIq[q1*lws + lid]) / sq[q1]);
            chi2[i] -= (SQUARE(fscale * tmpIq[q1 * lws + lid]) - 2 * targetIq[q1] * fscale * tmpIq[q1 * lws + lid]) / SQUARE(sq[q1]);
        }
        chi2[i] /= (float) nq;
    }
}

