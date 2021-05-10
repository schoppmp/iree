// RUN: iree-opt -split-input-file -iree-hal-link-executables -iree-hal-target-backends=vmvx %s | IreeFileCheck %s

module {
  hal.executable @dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vmvx, filter="vmvx" {
      hal.executable.entry_point @dispatch_0 attributes {interface = @io, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
      module {
        vm.module @module {
          vm.func @dispatch_0() {
            vm.return
          }
          vm.export @dispatch_0
        }
      }
    }
  }
  hal.executable @dispatch_1 attributes {sym_visibility = "private"} {
    hal.interface @io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vmvx, filter="vmvx" {
      hal.executable.entry_point @dispatch_1 attributes {interface = @io, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
      module {
        vm.module @module {
          vm.func @dispatch_1() {
            vm.return
          }
          vm.export @dispatch_1
        }
      }
    }
  }
  hal.executable @dispatch_2 attributes {sym_visibility = "private"} {
    hal.interface @io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @arg2, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vmvx, filter="vmvx" {
      hal.executable.entry_point @dispatch_2 attributes {interface = @io, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
      module {
        vm.module @module {
          vm.func @dispatch_2() {
            vm.return
          }
          vm.export @dispatch_2
        }
      }
    }
  }
  func @main() -> () {
    %device = hal.ex.shared_device : !hal.device
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
    %c1 = constant 1 : index
    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_0::@vmvx::@dispatch_0) workgroups([%c1, %c1, %c1])
    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_1::@vmvx::@dispatch_1) workgroups([%c1, %c1, %c1])
    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@dispatch_2::@vmvx::@dispatch_2) workgroups([%c1, %c1, %c1])
    return
  }
}

// All executables (including their interfaces and entry points) should be linked together into @linked_vmvx
// CHECK-NOT: hal.executable @dispatch_0
// CHECK-NOT: hal.executable @dispatch_1
// CHECK-NOT: hal.executable @dispatch_2
// CHECK:       hal.executable @vmvx_linked attributes {sym_visibility = "private"} {
// CHECK-NEXT:    hal.interface @io_0 {
// CHECK-NEXT:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// CHECK-NEXT:    }
// CHECK-NEXT:    hal.interface @io_1 {
// CHECK-NEXT:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @arg2, set=0, binding=1, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// CHECK-NEXT:    }
// CHECK-NEXT:    hal.executable.target @vmvx, filter="vmvx" {
// CHECK-NEXT:      hal.executable.entry_point @dispatch_0 attributes {interface = @io_0, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
// CHECK-NEXT:      hal.executable.entry_point @dispatch_1 attributes {interface = @io_0, ordinal = 1 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
// CHECK-NEXT:      hal.executable.entry_point @dispatch_2 attributes {interface = @io_1, ordinal = 2 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
// CHECK-NEXT:      module {
// CHECK-NEXT:        vm.module @linked_module {
// CHECK-NEXT:          vm.func @dispatch_0() {
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.export @dispatch_0
// CHECK-NEXT:          vm.func @dispatch_1() {
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.export @dispatch_1
// CHECK-NEXT:          vm.func @dispatch_2() {
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.export @dispatch_2
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
//
// CHECK:       func @main() {
// CHECK:         hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@vmvx_linked::@vmvx::@dispatch_2) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

module {
  hal.executable @dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vmvx, filter="vmvx" {
      hal.executable.entry_point @dispatch_0 attributes {interface = @io, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
      module {
        vm.module @module {
          vm.func @dispatch_0() {
            vm.return
          }
          vm.export @dispatch_0
        }
      }
    }
    hal.executable.target @othertarget, filter="othertarget" {
      module {
      }
    }
  }
  hal.executable @dispatch_1 attributes {sym_visibility = "private"} {
    hal.interface @io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vmvx, filter="vmvx" {
      hal.executable.entry_point @dispatch_1 attributes {interface = @io, ordinal = 0 : index, signature = (tensor<1x1xf32>) -> tensor<1x1xf32>}
      module {
        vm.module @module {
          vm.func @dispatch_1() {
            vm.return
          }
          vm.export @dispatch_1
        }
      }
    }
    hal.executable.target @othertarget, filter="othertarget" {
      module {
      }
    }
  }
  func @main() -> () {
    %device = hal.ex.shared_device : !hal.device
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot") categories("Transfer|Dispatch") : !hal.command_buffer
    hal.device.switch<%device : !hal.device>
    #hal.device.match.id<"vmvx">(%arg1 = %cmd : !hal.command_buffer) {
      %c1 = constant 1 : index
      hal.command_buffer.dispatch.symbol<%arg1 : !hal.command_buffer> target(@dispatch_0::@vmvx::@dispatch_0) workgroups([%c1, %c1, %c1])
      hal.command_buffer.dispatch.symbol<%arg1 : !hal.command_buffer> target(@dispatch_1::@vmvx::@dispatch_1) workgroups([%c1, %c1, %c1])
      hal.return
    },
    #hal.device.match.id<"othertarget">(%arg1 = %cmd : !hal.command_buffer) {
      %c1 = constant 1 : index
      hal.command_buffer.dispatch.symbol<%arg1 : !hal.command_buffer> target(@dispatch_0::@otherdispatch::@dispatch_0) workgroups([%c1, %c1, %c1])
      hal.command_buffer.dispatch.symbol<%arg1 : !hal.command_buffer> target(@dispatch_1::@otherdispatch::@dispatch_1) workgroups([%c1, %c1, %c1])
      hal.return
    }
    return
  }
}

// VMLA target should be pulled out from both executables
// CHECK:       hal.executable @vmvx_linked attributes {sym_visibility = "private"} {
// CHECK-NEXT:    hal.interface @io_0 {
// CHECK-NEXT:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// CHECK-NEXT:    }
// CHECK-NEXT:    hal.interface @io_1 {
// CHECK-NEXT:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// CHECK-NEXT:    }
// CHECK-NEXT:    hal.executable.target @vmvx, filter="vmvx" {
// CHECK-NEXT:      hal.executable.entry_point @dispatch_0 attributes {interface = @io_0, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
// CHECK-NEXT:      hal.executable.entry_point @dispatch_1 attributes {interface = @io_1, ordinal = 1 : index, signature = (tensor<1x1xf32>) -> tensor<1x1xf32>}
// CHECK-NEXT:      module {
// CHECK-NEXT:        vm.module @linked_module {
// CHECK-NEXT:          vm.func @dispatch_0() {
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.export @dispatch_0
// CHECK-NEXT:          vm.func @dispatch_1() {
// CHECK-NEXT:            vm.return
// CHECK-NEXT:          }
// CHECK-NEXT:          vm.export @dispatch_1
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
//
// @dispatch_0/1 should remain, with just @othertarget
// CHECK:  hal.executable @dispatch_0 attributes {sym_visibility = "private"} {
// CHECK:    hal.interface @io
// CHECK:    hal.executable.target @othertarget, filter="othertarget"
// CHECK:  hal.executable @dispatch_1 attributes {sym_visibility = "private"} {
// CHECK:    hal.interface @io
// CHECK:    hal.executable.target @othertarget, filter="othertarget"
//
// CHECK:       func @main() {
// CHECK:         hal.device.switch<%device : !hal.device>
// CHECK-NEXT:    #hal.device.match.id<"vmvx">(%arg0 = %cmd : !hal.command_buffer) {
// CHECK-NEXT:      %c1 = constant 1 : index
// CHECK-NEXT:      hal.command_buffer.dispatch.symbol<%arg0 : !hal.command_buffer> target(@vmvx_linked::@vmvx::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:      hal.command_buffer.dispatch.symbol<%arg0 : !hal.command_buffer> target(@vmvx_linked::@vmvx::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:      hal.return
// CHECK-NEXT:    },
// CHECK-NEXT:    #hal.device.match.id<"othertarget">(%arg0 = %cmd : !hal.command_buffer) {
// CHECK-NEXT:      %c1 = constant 1 : index
// CHECK-NEXT:      hal.command_buffer.dispatch.symbol<%arg0 : !hal.command_buffer> target(@dispatch_0::@otherdispatch::@dispatch_0) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:      hal.command_buffer.dispatch.symbol<%arg0 : !hal.command_buffer> target(@dispatch_1::@otherdispatch::@dispatch_1) workgroups([%c1, %c1, %c1])
// CHECK-NEXT:      hal.return
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// -----

module {
  hal.executable @dispatch_0 attributes {sym_visibility = "private"} {
    hal.interface @io {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vmvx, filter="vmvx" {
      hal.executable.entry_point @dispatch_0 attributes {interface = @io, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
      module {
        vm.module @module {}
      }
    }
  }
  hal.executable @dispatch_1 attributes {sym_visibility = "private"} {
    hal.interface @io attributes {push_constants = 2 : index} {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vmvx, filter="vmvx" {
      hal.executable.entry_point @dispatch_1 attributes {interface = @io, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
      module {
        vm.module @module {}
      }
    }
  }
  hal.executable @dispatch_2 attributes {sym_visibility = "private"} {
    hal.interface @io attributes {push_constants = 2 : index} {
      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    }
    hal.executable.target @vmvx, filter="vmvx" {
      hal.executable.entry_point @dispatch_2 attributes {interface = @io, ordinal = 0 : index, signature = (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>}
      module {
        vm.module @module {}
      }
    }
  }
}

// Interfaces with different numbers of push constants should remain separate.
// CHECK-NOT: hal.executable @dispatch_0
// CHECK-NOT: hal.executable @dispatch_1
// CHECK-NOT: hal.executable @dispatch_2
// CHECK:       hal.executable @vmvx_linked attributes {sym_visibility = "private"} {
// CHECK-NEXT:    hal.interface @io_0 {
// CHECK-NEXT:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// CHECK-NEXT:    }
// CHECK-NEXT:    hal.interface @io_1 attributes {push_constants = 2 : index} {
// CHECK-NEXT:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// CHECK-NEXT:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// CHECK-NEXT:    }
