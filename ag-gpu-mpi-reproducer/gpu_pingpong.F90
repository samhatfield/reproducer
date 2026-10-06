! GPU-aware MPI ping-pong reproducer.
!
! Two MPI ranks exchange messages whose send and receive buffers live in GPU
! device memory. Device buffers are managed exclusively through OpenACC, and
! device addresses are passed to MPI via HOST_DATA USE_DEVICE. If the MPI
! stack cannot handle device pointers (e.g. UCX without working CUDA
! transports), this program is expected to crash or report corrupted data.
!
! With --stage-via-host, each message is instead copied between device and
! host memory (!$acc update) around the MPI calls, so MPI only ever sees host
! buffers. This serves as a control that avoids direct GPU-GPU communication.
!
! Usage: mpirun -np 2 ./gpu_pingpong [--stage-via-host] [max_bytes] [iterations]

program gpu_pingpong
  use, intrinsic :: iso_fortran_env, only: real64, output_unit, error_unit
  use, intrinsic :: iso_c_binding, only: c_loc, c_intptr_t
  use mpi_f08
  use openacc
  implicit none

  integer, parameter :: default_max_bytes = 64 * 1024 * 1024
  integer, parameter :: default_iterations = 10
  integer, parameter :: tag = 42

  real(real64), allocatable, target :: sbuf(:), rbuf(:)
  integer :: rank, nranks, peer, local_rank, ndevices, device_num
  integer :: max_bytes, iterations, max_count, count, iter, nerrors, total_errors
  type(MPI_Comm) :: local_comm
  type(MPI_Status) :: status
  real(real64) :: t0, elapsed
  logical :: stage_via_host

  call MPI_Init()
  call MPI_Comm_rank(MPI_COMM_WORLD, rank)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks)

  if (nranks /= 2) then
    if (rank == 0) write(error_unit, '(a,i0)') 'ERROR: this program needs exactly 2 MPI ranks, got ', nranks
    call MPI_Abort(MPI_COMM_WORLD, 1)
  end if
  peer = 1 - rank

  call parse_arguments(max_bytes, iterations, stage_via_host)
  max_count = max(1, max_bytes / (storage_size(1.0_real64) / 8))

  ! Bind each rank on a node to its own GPU (round-robin)
  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, local_comm)
  call MPI_Comm_rank(local_comm, local_rank)
  call MPI_Comm_free(local_comm)
  ndevices = acc_get_num_devices(acc_device_nvidia)
  if (ndevices < 1) then
    write(error_unit, '(a,i0)') 'ERROR: no NVIDIA devices visible to rank ', rank
    call MPI_Abort(MPI_COMM_WORLD, 1)
  end if
  device_num = mod(local_rank, ndevices)
  call acc_set_device_num(device_num, acc_device_nvidia)
  call acc_init(acc_device_nvidia)

  allocate(sbuf(max_count), rbuf(max_count))
  !$acc enter data create(sbuf, rbuf)

  call report_setup()

  if (rank == 0) then
    write(output_unit, '(/,a12,a12,a16,a14,a10)') 'bytes', 'count', 'avg RTT (us)', 'BW (MB/s)', 'status'
    flush(output_unit)
  end if

  total_errors = 0
  count = 1
  do while (count <= max_count)
    call fill_send_buffer(count)

    call MPI_Barrier(MPI_COMM_WORLD)
    t0 = MPI_Wtime()

    do iter = 1, iterations
      if (stage_via_host) then
        call pingpong_via_host(count)
      else
        call pingpong_device(count)
      end if
    end do

    elapsed = MPI_Wtime() - t0

    ! Both ranks validate what they last received against rank 0's pattern
    nerrors = check_receive_buffer(count)
    call MPI_Allreduce(MPI_IN_PLACE, nerrors, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD)
    total_errors = total_errors + nerrors

    if (rank == 0) then
      write(output_unit, '(i12,i12,f16.2,f14.1,a10)') &
        count * 8, count, 1.0e6_real64 * elapsed / iterations, &
        2.0_real64 * iterations * count * 8 / elapsed / 1.0e6_real64, &
        merge('OK     ', 'CORRUPT', nerrors == 0)
      flush(output_unit)
    end if

    count = count * 2
  end do

  !$acc exit data delete(sbuf, rbuf)
  deallocate(sbuf, rbuf)

  if (rank == 0) then
    if (total_errors == 0) then
      write(output_unit, '(/,a)') 'PASSED: all transfers completed with correct data'
    else
      write(output_unit, '(/,a,i0,a)') 'FAILED: ', total_errors, ' corrupted elements detected'
    end if
  end if

  call MPI_Finalize()

contains

  ! Pass device addresses straight to MPI (requires GPU-aware MPI)
  subroutine pingpong_device(count)
    integer, intent(in) :: count

    !$acc host_data use_device(sbuf, rbuf)
    if (rank == 0) then
      call MPI_Send(sbuf, count, MPI_DOUBLE_PRECISION, peer, tag, MPI_COMM_WORLD)
      call MPI_Recv(rbuf, count, MPI_DOUBLE_PRECISION, peer, tag, MPI_COMM_WORLD, status)
    else
      call MPI_Recv(rbuf, count, MPI_DOUBLE_PRECISION, peer, tag, MPI_COMM_WORLD, status)
      call MPI_Send(rbuf, count, MPI_DOUBLE_PRECISION, peer, tag, MPI_COMM_WORLD)
    end if
    !$acc end host_data
  end subroutine pingpong_device

  ! Copy send data device->host before MPI_Send and received data host->device
  ! after MPI_Recv, so MPI only ever touches host memory
  subroutine pingpong_via_host(count)
    integer, intent(in) :: count

    if (rank == 0) then
      !$acc update self(sbuf(1:count))
      call MPI_Send(sbuf, count, MPI_DOUBLE_PRECISION, peer, tag, MPI_COMM_WORLD)
      call MPI_Recv(rbuf, count, MPI_DOUBLE_PRECISION, peer, tag, MPI_COMM_WORLD, status)
      !$acc update device(rbuf(1:count))
    else
      call MPI_Recv(rbuf, count, MPI_DOUBLE_PRECISION, peer, tag, MPI_COMM_WORLD, status)
      !$acc update device(rbuf(1:count))
      !$acc update self(rbuf(1:count))
      call MPI_Send(rbuf, count, MPI_DOUBLE_PRECISION, peer, tag, MPI_COMM_WORLD)
    end if
  end subroutine pingpong_via_host

  subroutine parse_arguments(max_bytes, iterations, stage_via_host)
    integer, intent(out) :: max_bytes, iterations
    logical, intent(out) :: stage_via_host
    character(len=256) :: arg
    integer :: i, ios, npositional

    max_bytes = default_max_bytes
    iterations = default_iterations
    stage_via_host = .false.
    npositional = 0

    do i = 1, command_argument_count()
      call get_command_argument(i, arg)
      select case (arg)
      case ('--stage-via-host')
        stage_via_host = .true.
      case ('-h', '--help')
        call print_usage()
        call MPI_Finalize()
        stop
      case default
        if (arg(1:1) == '-') then
          if (rank == 0) write(error_unit, '(a)') 'ERROR: unknown option ' // trim(arg)
          call print_usage()
          call MPI_Abort(MPI_COMM_WORLD, 1)
        end if
        npositional = npositional + 1
        select case (npositional)
        case (1)
          read(arg, *, iostat=ios) max_bytes
          if (ios /= 0 .or. max_bytes < 8) max_bytes = default_max_bytes
        case (2)
          read(arg, *, iostat=ios) iterations
          if (ios /= 0 .or. iterations < 1) iterations = default_iterations
        end select
      end select
    end do
  end subroutine parse_arguments

  subroutine print_usage()
    if (rank /= 0) return
    write(output_unit, '(a)') 'Usage: mpirun -np 2 gpu_pingpong [--stage-via-host] [max_bytes] [iterations]', &
      '  --stage-via-host  copy buffers to/from host around MPI calls (no GPU-direct)', &
      '  max_bytes         largest message size in bytes (default 67108864)', &
      '  iterations        round trips per message size (default 10)'
    flush(output_unit)
  end subroutine print_usage

  ! Print MPI/GPU configuration and confirm the buffers really are distinct
  ! device allocations (i.e. not host or unified memory)
  subroutine report_setup()
    character(len=MPI_MAX_LIBRARY_VERSION_STRING) :: version
    character(len=MPI_MAX_PROCESSOR_NAME) :: host
    integer :: length, r
    integer(c_intptr_t) :: host_addr, dev_addr

    if (rank == 0) then
      call MPI_Get_library_version(version, length)
      write(output_unit, '(a)') 'MPI library: ' // trim(version(1:length))
      write(output_unit, '(a,i0,a,i0,a,i0,a)') 'Message sizes: 8 .. ', max_count * 8, &
        ' bytes, ', iterations, ' round trips per size'
      if (stage_via_host) then
        write(output_unit, '(a)') 'Mode: staged via host (MPI is given host buffers)'
      else
        write(output_unit, '(a)') 'Mode: GPU-direct (MPI is given device buffers)'
      end if
    end if
    call MPI_Get_processor_name(host, length)

    host_addr = transfer(c_loc(sbuf), host_addr)
    dev_addr = transfer(acc_deviceptr(sbuf), dev_addr)

    do r = 0, nranks - 1
      if (r == rank) then
        write(output_unit, '(a,i0,a,a,a,i0,a,i0,a,l1,a,z0,a,z0)') &
          'rank ', rank, ' on ', trim(host(1:length)), ': GPU ', device_num, ' of ', ndevices, &
          ', sbuf present=', acc_is_present(sbuf), ', host addr=0x', host_addr, ', device addr=0x', dev_addr
        if (host_addr == dev_addr) then
          write(output_unit, '(a,i0,a)') 'WARNING: rank ', rank, &
            ' host and device addresses coincide; buffers may not be in discrete device memory'
        end if
        flush(output_unit)
      end if
      call MPI_Barrier(MPI_COMM_WORLD)
    end do
  end subroutine report_setup

  pure function pattern(i, count) result(val)
    integer, intent(in) :: i, count
    real(real64) :: val
    !$acc routine seq
    val = real(i, real64) + 1.0e-3_real64 * real(count, real64)
  end function pattern

  ! Rank 0 fills its send buffer with a known pattern; both ranks poison
  ! their receive buffers so stale data cannot pass the check
  subroutine fill_send_buffer(count)
    integer, intent(in) :: count
    integer :: i

    !$acc parallel loop present(sbuf, rbuf)
    do i = 1, count
      sbuf(i) = merge(pattern(i, count), -1.0_real64, rank == 0)
      rbuf(i) = -huge(1.0_real64)
    end do
  end subroutine fill_send_buffer

  integer function check_receive_buffer(count) result(nerr)
    integer, intent(in) :: count
    integer :: i

    nerr = 0
    !$acc parallel loop present(rbuf) reduction(+:nerr)
    do i = 1, count
      if (rbuf(i) /= pattern(i, count)) nerr = nerr + 1
    end do
  end function check_receive_buffer

end program gpu_pingpong
