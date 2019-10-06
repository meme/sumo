from typing import List
from binaryninja import *
from llvmlite import ir
from llvmlite import binding as llvm

from ctypes import CFUNCTYPE, c_int64, c_int

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

def ninja_to_ir_type(t: Type):
    # TODO(keegan) check if values are signed or unsigned
    if t.type_class == TypeClass.IntegerTypeClass:
       return ir.IntType(t.width * 8)
    else:
        raise ValueError()


# TODO(keegan) check if function can return
# TODO(keegan) check if function has variable arguments
def ir_function_type(f: Function) -> ir.FunctionType:
    return_type: Type = f.function_type.return_value
    parameters: List[FunctionParameter] = f.function_type.parameters

    ir_return_type = ninja_to_ir_type(return_type)
    ir_parameters_type = list(map(lambda x: ninja_to_ir_type(x.type), parameters))

    return ir.FunctionType(ir_return_type, ir_parameters_type)


def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine


def compile_ir(engine, mod: llvm.ModuleRef):
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


SSAVariable.__hash__ = lambda self: hash(str(self.var) + "#" + str(self.version))


# https://github.com/joshwatson/f-ing-around-with-binaryninja/blob/master/ep4-emulator/vm_visitor.py#L4
class BNILVisitor(object):
    def __init__(self, **kw):
        super(BNILVisitor, self).__init__()

    def visit(self, expression):
        method_name = 'visit_{}'.format(expression.operation.name)
        if hasattr(self, method_name):
            value = getattr(self, method_name)(expression)
        else:
            print(method_name + ' undefined ')
            value = None
        return value


class QueuedIncomingPhi:
    """A phi instruction ready for incoming node insertion"""
    def __init__(self, phi: ir.PhiInstr, var: str, ir_block: ir.Block):
        self.phi = phi
        self.var = var
        self.ir_block = ir_block


class FunctionLifter:
    """Lift a MediumLevelILFunction to a LLVM IR function"""
    def __init__(self, ir_module: ir.Module, f: Function):
        self.ir_basic_blocks = {}
        self.phis: List[QueuedIncomingPhi] = []
        self.variables = {}
        self.f = f

        self.ir_module = ir_module
        self.ir_function = ir.Function(self.ir_module, ir_function_type(self.f), name=self.f.name)

        self.pass_manager = llvm.create_module_pass_manager()
        self.pass_manager_b = llvm.create_pass_manager_builder()

    def run(self):
        for basic_block in self.f.medium_level_il.ssa_form.basic_blocks:
            self.ir_basic_blocks[basic_block.index] = self.ir_function.append_basic_block()

        self.ir_basic_blocks[0].name = "entry"

        for basic_block in self.f.medium_level_il.ssa_form.basic_blocks:
            v = LifterVisitor(self, basic_block.index, basic_block)
            for instr in basic_block:
                v.visit(instr)

        for entry in self.phis:
            phi: ir.PhiInstr = entry.phi
            phi.add_incoming(self.variables[entry.var], entry.ir_block)

    def optimize(self, level: int) -> llvm.ModuleRef:
        """Optimize the produced LLVM IR, 3 corresponds to -O3, etc."""
        opt_module = llvm.parse_assembly(str(self.ir_module))
        self.pass_manager_b = llvm.create_pass_manager_builder()
        self.pass_manager_b.opt_level = level
        self.pass_manager_b.populate(self.pass_manager)
        self.pass_manager.run(opt_module)
        return opt_module


class LifterVisitor(BNILVisitor):
    def __init__(self, function_lifter: FunctionLifter, basic_block_index: int, basic_block: MediumLevelILBasicBlock):
        super(LifterVisitor, self).__init__()
        self.ir_basic_blocks = function_lifter.ir_basic_blocks
        self.ir_basic_block = function_lifter.ir_basic_blocks[basic_block_index]
        self.basic_block = basic_block
        self.builder = ir.IRBuilder(self.ir_basic_block)
        self.f = function_lifter.f
        self.variables = function_lifter.variables
        self.phis = function_lifter.phis
        self.ir_function = function_lifter.ir_function

    def ir_bb_for_ssa_var(self, var: SSAVariable) -> ir.Block:
        """Return the IR basic block for which an SSAVariable was defined"""
        basic_block: MediumLevelILBasicBlock = f.medium_level_il.get_ssa_var_definition(var).il_basic_block
        return self.ir_basic_blocks[basic_block.index]

    def ir_bb_for_instr(self, instr_index: int) -> ir.Block:
        """Return the IR basic block which contains the MLIL instruction"""
        return self.ir_basic_blocks[self.f.mlil[instr_index].il_basic_block.index]

    def visit_MLIL_SET_VAR_SSA(self, expr):
        self.variables[expr.dest] = self.visit(expr.src)

    def visit_MLIL_VAR_SSA_FIELD(self, expr):
        # FIXME(keegan) this doesn't pull function arg properly

        src = None
        if expr.src.var.name == "arg1":
            src = self.ir_function.args[0]
        else:
            if expr.src in self.variables:
                src = self.variables[expr.src]
            else:
                raise ValueError()

        # FIXME(keegan) not always an integer type
        # FIXME(keegan) truncating does not work for structure offsets, how to do this in LLVM?
        if expr.offset != 0:
            raise ValueError('structure offsets not supported yet')

        return self.builder.trunc(src, ir.IntType(expr.size * 8))

    def visit_MLIL_CONST(self, expr):
        return ir.Constant(ninja_to_ir_type(expr.expr_type), expr.value.value)

    def visit_MLIL_VAR_SSA(self, expr):
        if expr.src in self.variables:
            return self.variables[expr.src]
        else:
            raise ValueError('asking for variable before definition')

    def visit_MLIL_VAR_PHI(self, expr):
        dest, srcs = expr.operands

        phi = self.builder.phi(ninja_to_ir_type(expr.dest.var.type))
        self.variables[dest] = phi
        for src in srcs:
            self.phis.append(QueuedIncomingPhi(phi, src, self.ir_bb_for_ssa_var(src)))
        return phi

    def visit_MLIL_AND(self, expr):
        lhs, rhs = expr.operands
        return self.builder.and_(self.visit(lhs), self.visit(rhs))

    def visit_MLIL_ZX(self, expr):
        return self.builder.zext(self.visit(expr.src), ninja_to_ir_type(expr.expr_type))

    def visit_MLIL_ADD(self, expr):
        lhs, rhs = expr.operands
        return self.builder.add(self.visit(lhs), self.visit(rhs))

    def visit_MLIL_MUL(self, expr):
        lhs, rhs = expr.operands
        return self.builder.mul(self.visit(lhs), self.visit(rhs))

    def visit_MLIL_XOR(self, expr):
        lhs, rhs = expr.operands
        return self.builder.xor(self.visit(lhs), self.visit(rhs))

    def visit_MLIL_OR(self, expr):
        lhs, rhs = expr.operands
        return self.builder.or_(self.visit(lhs), self.visit(rhs))

    def visit_MLIL_GOTO(self, expr):
        return self.builder.branch(self.ir_bb_for_instr(expr.dest))

    def visit_MLIL_LOW_PART(self, expr):
        return self.builder.trunc(self.visit(expr.src), ir.IntType(expr.size * 8))

    def visit_MLIL_CMP_NE(self, expr):
        lhs, rhs = expr.operands
        return self.builder.icmp_unsigned('!=', self.visit(lhs), self.visit(rhs))

    def visit_MLIL_IF(self, expr):
        condition = expr.condition
        true_branch = self.ir_bb_for_instr(expr.true)
        false_branch = self.ir_bb_for_instr(expr.false)
        return self.builder.cbranch(self.visit(condition), true_branch, false_branch)

    def visit_MLIL_RET(self, expr):
        if len(expr.src) > 1:
            raise ValueError('multiple return types not supported')

        return self.builder.ret(self.visit(expr.src[0]))


def main(bv: BinaryView):
    # Lift the `target` function to IR
    module = ir.Module(name=__file__)
    f: Function = bv.get_function_at(bv.get_symbols_by_name('target')[0].address)
    lifter = FunctionLifter(module, f)
    lifter.run()

    # Output the optimized IR to a CFG
    opt_module: llvm.ModuleRef = lifter.optimize(3)
    opt_target = opt_module.get_function(f.name)
    dot = llvm.get_function_cfg(opt_target)
    open('output.opt.dot', 'w').write(dot)

    # Execute the LLVM IR
    engine = create_execution_engine()
    mod = compile_ir(engine, opt_module)
    func_ptr = engine.get_function_address("target_0")
    cfunc = CFUNCTYPE(c_int64, c_int)(func_ptr)
    print(cfunc(10))