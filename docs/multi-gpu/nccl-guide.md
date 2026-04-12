# NCCL (NVIDIA Collective Communications Library) Reference

## Overview

NCCL provides optimized collective communication for multi-GPU and multi-node systems. It auto-selects the best paths (NVLink, PCIe, InfiniBand, sockets). All operations are async, integrated with CUDA streams.

## Communicator Setup

```c
// Step 1: Generate unique ID (one rank only)
ncclUniqueId id;
ncclGetUniqueId(&id);
// Broadcast id to all ranks via MPI, sockets, etc.

// Step 2: Each rank creates communicator
ncclComm_t comm;
ncclCommInitRank(&comm, nranks, id, rank);

// Cleanup
ncclCommFinalize(comm);
ncclCommDestroy(comm);
```

Single-process multi-GPU convenience:
```c
ncclComm_t comms[ndev];
ncclCommInitAll(comms, ndev, devlist);  // devlist=NULL uses devices 0..ndev-1
```

### With config (nonblocking, CGA clusters, etc.)
```c
ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
config.blocking = 0;          // nonblocking mode
config.cgaClusterSize = 4;    // CGA cluster size (sm90+)
ncclCommInitRankConfig(&comm, nranks, id, rank, &config);
```

### Split (like MPI_Comm_split)
```c
ncclCommSplit(comm, color, key, &newcomm, &config);
// Same color → same sub-communicator. key = ordering within new comm.
```

## Data Types

`ncclInt8`/`ncclChar`, `ncclUint8`, `ncclInt32`/`ncclInt`, `ncclUint32`, `ncclInt64`, `ncclUint64`, `ncclFloat16`/`ncclHalf`, `ncclFloat32`/`ncclFloat`, `ncclFloat64`/`ncclDouble`, `ncclBfloat16`, `ncclFloat8e4m3`, `ncclFloat8e5m2`

## Reduction Operations

`ncclSum`, `ncclProd`, `ncclMax`, `ncclMin`, `ncclAvg`

## Collective Operations

All take `(sendbuff, recvbuff, count, datatype, ..., comm, stream)` and return `ncclResult_t`. Operations are enqueued to the stream and return immediately.

### AllReduce
```c
ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
// out[i] = op(in_0[i], ..., in_{k-1}[i]) on ALL ranks
// In-place: sendbuff == recvbuff
```

### Broadcast
```c
ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
// Copies from root's sendbuff to all ranks' recvbuff
```

### Reduce
```c
ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
// Reduces to root's recvbuff only
```

### AllGather
```c
ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
// Each rank contributes sendcount elements
// recvbuff has nranks * sendcount elements, rank i at offset i * sendcount
// In-place: sendbuff == recvbuff + rank * sendcount
```

### ReduceScatter
```c
ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
// Reduce then scatter: rank i gets i-th block of recvcount elements
// sendbuff has nranks * recvcount elements
// In-place: recvbuff == sendbuff + rank * recvcount
```

### AlltoAll
```c
ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream);
// Each rank sends count elements to every other rank
// Data to rank j from sendbuff + j*count; from rank i at recvbuff + i*count
// In-place NOT supported
```

### Gather / Scatter
```c
ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream);
ncclScatter(sendbuff, recvbuff, count, datatype, root, comm, stream);
```

## Point-to-Point Operations

```c
ncclSend(sendbuff, count, datatype, peer, comm, stream);
ncclRecv(recvbuff, count, datatype, peer, comm, stream);
```

**Must be grouped** for concurrent progress:
```c
ncclGroupStart();
ncclSend(sendbuf, count, type, peer, comm, stream);
ncclRecv(recvbuf, count, type, peer, comm, stream);
ncclGroupEnd();
```

## One-Sided RMA (NCCL 2.29+)

```c
// Register window
ncclCommWindowRegister(comm, buff, size, &win, flags);
// flags: NCCL_WIN_COLL_SYMMETRIC, NCCL_WIN_STRICT_ORDERING

// Put data + signal
ncclPutSignal(localbuff, count, datatype, peer, peerWin, offset, sigIdx, ctx, flags, comm, stream);

// Signal without data
ncclSignal(peer, sigIdx, ctx, flags, comm, stream);

// Wait for signals
ncclWaitSignal(nDesc, signalDescs, comm, stream);

// Cleanup
ncclCommWindowDeregister(comm, win);
```

## Group API

```c
ncclGroupStart();
// ... multiple NCCL calls ...
ncclGroupEnd();  // fuses all into single launch
```

**Three use cases:**
1. **Multi-GPU from one thread** — prevents deadlock
2. **Operation fusion** — reduces launch overhead
3. **Concurrent send/recv** — enables progress

**Rules:**
- All ranks must issue operations in identical order
- Groups can be nested; only outermost `ncclGroupEnd()` triggers execution
- In nonblocking mode, `ncclGroupEnd()` may return `ncclInProgress` — poll `ncclCommGetAsyncError()`

## Stream Semantics

- Operations enqueue to the given `cudaStream_t` and return immediately
- Completion via `cudaStreamSynchronize()` or CUDA events
- Within a group using multiple streams, NCCL creates inter-stream dependencies (global sync barrier)

## In-Place Rules

| Operation | In-place condition |
|---|---|
| AllReduce | `sendbuff == recvbuff` |
| Broadcast | `sendbuff == recvbuff` |
| Reduce | `sendbuff == recvbuff` |
| AllGather | `sendbuff == recvbuff + rank * sendcount` |
| ReduceScatter | `recvbuff == sendbuff + rank * recvcount` |
| AlltoAll | **NOT supported** |

## Key Environment Variables

### Algorithm / Protocol
| Variable | Values | Default | Description |
|---|---|---|---|
| `NCCL_ALGO` | Ring, Tree, Collnet, NVLS, NVLSTree, PAT | auto | Allowed algorithms |
| `NCCL_PROTO` | LL, LL128, Simple | all | Communication protocols |

Use `^` to exclude: `NCCL_ALGO=^Collnet`. Function-specific: `NCCL_ALGO="AllReduce:Ring,Tree;Broadcast:Ring"`

### Network
| Variable | Default | Description |
|---|---|---|
| `NCCL_NET` | auto | Force transport: `IB`, `Socket` |
| `NCCL_SOCKET_IFNAME` | auto | Interface selection (prefix, `=exact`, `^exclude`) |
| `NCCL_IB_DISABLE` | 0 | Force sockets over IB |
| `NCCL_IB_HCA` | auto | Select HCA interfaces |
| `NCCL_IB_QPS_PER_CONNECTION` | 1 | Queue pairs (1-128) |
| `NCCL_IB_ADAPTIVE_ROUTING` | 1 (IB) | Enable adaptive routing |
| `NCCL_CROSS_NIC` | 2 (auto) | Cross-NIC communication |

### Performance
| Variable | Default | Description |
|---|---|---|
| `NCCL_BUFFSIZE` | 4194304 | Buffer size (bytes, power of 2) |
| `NCCL_NTHREADS` | 512 | CUDA threads per NCCL block |
| `NCCL_MAX_CTAS` | auto | Max communication CTAs (1-64) |
| `NCCL_MIN_CTAS` | auto | Min communication CTAs (1-64) |
| `NCCL_NVLS_ENABLE` | 2 (auto) | NVLink SHARP (Hopper+) |
| `NCCL_COLLNET_ENABLE` | 0 | In-network reduction |
| `NCCL_P2P_DISABLE` | 0 | Disable peer-to-peer |
| `NCCL_SHM_DISABLE` | 0 | Disable shared memory transport |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | 0 | Implicit ordering across communicators |

### GPU Direct RDMA
| Variable | Default | Description |
|---|---|---|
| `NCCL_NET_GDR_LEVEL` | auto | Max GDR distance (LOC, PIX, PXB, PHB, SYS) |
| `NCCL_NET_GDR_READ` | 1 (NVLink) | GDR for send operations |
| `NCCL_DMABUF_ENABLE` | 1 | GDR via dma-buf |

### Debugging
| Variable | Values | Description |
|---|---|---|
| `NCCL_DEBUG` | VERSION, WARN, INFO, TRACE | Debug output level |
| `NCCL_DEBUG_FILE` | filename | Log to file (`%h`=host, `%p`=PID) |
| `NCCL_DEBUG_SUBSYS` | INIT, COLL, P2P, NET, GRAPH, TUNING, ALL | Filter subsystems |

## Performance Tuning

1. **Algorithm**: Let NCCL auto-select. Ring = large messages, Tree = small/latency-sensitive, NVLS/NVLSTree = NVSwitch (Hopper+).
2. **Protocol**: Simple = large messages (high BW), LL = small messages (low latency), LL128 = medium.
3. **CTAs**: More `NCCL_MAX_CTAS` = more throughput but fewer SMs for compute. Balance based on overlap needs.
4. **Buffer registration**: `ncclCommRegister()` / `ncclMemAlloc()` for frequent buffers → zero-copy.
5. **Operation fusion**: Group small collectives in `ncclGroupStart/End` to amortize overhead.
6. **Stream overlap**: Dedicate streams for NCCL, overlap with compute on other streams.
7. **CUDA Graphs**: Capture repetitive patterns to eliminate host overhead.

## Error Handling

```c
ncclResult_t err = ncclAllReduce(...);
if (err != ncclSuccess) {
    printf("NCCL error: %s\n", ncclGetErrorString(err));
    printf("Detail: %s\n", ncclGetLastError(comm));
    // For system/remote errors: must abort and recreate
    ncclCommAbort(comm);
}

// For async errors (nonblocking mode):
ncclResult_t asyncErr;
ncclCommGetAsyncError(comm, &asyncErr);
```

| Error | Action |
|---|---|
| `ncclInvalidArgument` | Fix arguments, retry |
| `ncclInvalidUsage` | Fix usage pattern |
| `ncclUnhandledCudaError` / `ncclSystemError` / `ncclRemoteError` | `ncclCommAbort()` + recreate |
| `ncclInternalError` | NCCL bug — `ncclCommAbort()` |

## Thread Safety

- NCCL is **NOT thread-safe** but IS **reentrant**
- Do NOT call NCCL from multiple threads on the same communicator concurrently
- All operations in a group must come from a single thread
- Different communicators on different devices CAN be used from different threads (with care)
