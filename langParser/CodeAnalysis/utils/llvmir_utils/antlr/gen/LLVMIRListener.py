# Generated from LLVMIR.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .LLVMIRParser import LLVMIRParser
else:
    from LLVMIRParser import LLVMIRParser

# This class defines a complete listener for a parse tree produced by LLVMIRParser.
class LLVMIRListener(ParseTreeListener):

    # Enter a parse tree produced by LLVMIRParser#compilationUnit.
    def enterCompilationUnit(self, ctx:LLVMIRParser.CompilationUnitContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#compilationUnit.
    def exitCompilationUnit(self, ctx:LLVMIRParser.CompilationUnitContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#targetDef.
    def enterTargetDef(self, ctx:LLVMIRParser.TargetDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#targetDef.
    def exitTargetDef(self, ctx:LLVMIRParser.TargetDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sourceFilename.
    def enterSourceFilename(self, ctx:LLVMIRParser.SourceFilenameContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sourceFilename.
    def exitSourceFilename(self, ctx:LLVMIRParser.SourceFilenameContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#targetDataLayout.
    def enterTargetDataLayout(self, ctx:LLVMIRParser.TargetDataLayoutContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#targetDataLayout.
    def exitTargetDataLayout(self, ctx:LLVMIRParser.TargetDataLayoutContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#targetTriple.
    def enterTargetTriple(self, ctx:LLVMIRParser.TargetTripleContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#targetTriple.
    def exitTargetTriple(self, ctx:LLVMIRParser.TargetTripleContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#topLevelEntity.
    def enterTopLevelEntity(self, ctx:LLVMIRParser.TopLevelEntityContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#topLevelEntity.
    def exitTopLevelEntity(self, ctx:LLVMIRParser.TopLevelEntityContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#moduleAsm.
    def enterModuleAsm(self, ctx:LLVMIRParser.ModuleAsmContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#moduleAsm.
    def exitModuleAsm(self, ctx:LLVMIRParser.ModuleAsmContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#typeDef.
    def enterTypeDef(self, ctx:LLVMIRParser.TypeDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#typeDef.
    def exitTypeDef(self, ctx:LLVMIRParser.TypeDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#comdatDef.
    def enterComdatDef(self, ctx:LLVMIRParser.ComdatDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#comdatDef.
    def exitComdatDef(self, ctx:LLVMIRParser.ComdatDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#globalDecl.
    def enterGlobalDecl(self, ctx:LLVMIRParser.GlobalDeclContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#globalDecl.
    def exitGlobalDecl(self, ctx:LLVMIRParser.GlobalDeclContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#globalDef.
    def enterGlobalDef(self, ctx:LLVMIRParser.GlobalDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#globalDef.
    def exitGlobalDef(self, ctx:LLVMIRParser.GlobalDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#indirectSymbolDef.
    def enterIndirectSymbolDef(self, ctx:LLVMIRParser.IndirectSymbolDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#indirectSymbolDef.
    def exitIndirectSymbolDef(self, ctx:LLVMIRParser.IndirectSymbolDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#funcDecl.
    def enterFuncDecl(self, ctx:LLVMIRParser.FuncDeclContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#funcDecl.
    def exitFuncDecl(self, ctx:LLVMIRParser.FuncDeclContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#funcDef.
    def enterFuncDef(self, ctx:LLVMIRParser.FuncDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#funcDef.
    def exitFuncDef(self, ctx:LLVMIRParser.FuncDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#attrGroupDef.
    def enterAttrGroupDef(self, ctx:LLVMIRParser.AttrGroupDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#attrGroupDef.
    def exitAttrGroupDef(self, ctx:LLVMIRParser.AttrGroupDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#namedMetadataDef.
    def enterNamedMetadataDef(self, ctx:LLVMIRParser.NamedMetadataDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#namedMetadataDef.
    def exitNamedMetadataDef(self, ctx:LLVMIRParser.NamedMetadataDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#metadataDef.
    def enterMetadataDef(self, ctx:LLVMIRParser.MetadataDefContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#metadataDef.
    def exitMetadataDef(self, ctx:LLVMIRParser.MetadataDefContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#useListOrder.
    def enterUseListOrder(self, ctx:LLVMIRParser.UseListOrderContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#useListOrder.
    def exitUseListOrder(self, ctx:LLVMIRParser.UseListOrderContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#useListOrderBB.
    def enterUseListOrderBB(self, ctx:LLVMIRParser.UseListOrderBBContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#useListOrderBB.
    def exitUseListOrderBB(self, ctx:LLVMIRParser.UseListOrderBBContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#funcHeader.
    def enterFuncHeader(self, ctx:LLVMIRParser.FuncHeaderContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#funcHeader.
    def exitFuncHeader(self, ctx:LLVMIRParser.FuncHeaderContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#indirectSymbol.
    def enterIndirectSymbol(self, ctx:LLVMIRParser.IndirectSymbolContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#indirectSymbol.
    def exitIndirectSymbol(self, ctx:LLVMIRParser.IndirectSymbolContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#callingConv.
    def enterCallingConv(self, ctx:LLVMIRParser.CallingConvContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#callingConv.
    def exitCallingConv(self, ctx:LLVMIRParser.CallingConvContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#callingConvInt.
    def enterCallingConvInt(self, ctx:LLVMIRParser.CallingConvIntContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#callingConvInt.
    def exitCallingConvInt(self, ctx:LLVMIRParser.CallingConvIntContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#funcHdrField.
    def enterFuncHdrField(self, ctx:LLVMIRParser.FuncHdrFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#funcHdrField.
    def exitFuncHdrField(self, ctx:LLVMIRParser.FuncHdrFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#gc.
    def enterGc(self, ctx:LLVMIRParser.GcContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#gc.
    def exitGc(self, ctx:LLVMIRParser.GcContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#prefix.
    def enterPrefix(self, ctx:LLVMIRParser.PrefixContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#prefix.
    def exitPrefix(self, ctx:LLVMIRParser.PrefixContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#prologue.
    def enterPrologue(self, ctx:LLVMIRParser.PrologueContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#prologue.
    def exitPrologue(self, ctx:LLVMIRParser.PrologueContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#personality.
    def enterPersonality(self, ctx:LLVMIRParser.PersonalityContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#personality.
    def exitPersonality(self, ctx:LLVMIRParser.PersonalityContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#returnAttribute.
    def enterReturnAttribute(self, ctx:LLVMIRParser.ReturnAttributeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#returnAttribute.
    def exitReturnAttribute(self, ctx:LLVMIRParser.ReturnAttributeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#funcBody.
    def enterFuncBody(self, ctx:LLVMIRParser.FuncBodyContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#funcBody.
    def exitFuncBody(self, ctx:LLVMIRParser.FuncBodyContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#basicBlock.
    def enterBasicBlock(self, ctx:LLVMIRParser.BasicBlockContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#basicBlock.
    def exitBasicBlock(self, ctx:LLVMIRParser.BasicBlockContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#instruction.
    def enterInstruction(self, ctx:LLVMIRParser.InstructionContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#instruction.
    def exitInstruction(self, ctx:LLVMIRParser.InstructionContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#terminator.
    def enterTerminator(self, ctx:LLVMIRParser.TerminatorContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#terminator.
    def exitTerminator(self, ctx:LLVMIRParser.TerminatorContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#localDefTerm.
    def enterLocalDefTerm(self, ctx:LLVMIRParser.LocalDefTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#localDefTerm.
    def exitLocalDefTerm(self, ctx:LLVMIRParser.LocalDefTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#valueTerminator.
    def enterValueTerminator(self, ctx:LLVMIRParser.ValueTerminatorContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#valueTerminator.
    def exitValueTerminator(self, ctx:LLVMIRParser.ValueTerminatorContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#retTerm.
    def enterRetTerm(self, ctx:LLVMIRParser.RetTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#retTerm.
    def exitRetTerm(self, ctx:LLVMIRParser.RetTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#brTerm.
    def enterBrTerm(self, ctx:LLVMIRParser.BrTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#brTerm.
    def exitBrTerm(self, ctx:LLVMIRParser.BrTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#condBrTerm.
    def enterCondBrTerm(self, ctx:LLVMIRParser.CondBrTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#condBrTerm.
    def exitCondBrTerm(self, ctx:LLVMIRParser.CondBrTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#switchTerm.
    def enterSwitchTerm(self, ctx:LLVMIRParser.SwitchTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#switchTerm.
    def exitSwitchTerm(self, ctx:LLVMIRParser.SwitchTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#indirectBrTerm.
    def enterIndirectBrTerm(self, ctx:LLVMIRParser.IndirectBrTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#indirectBrTerm.
    def exitIndirectBrTerm(self, ctx:LLVMIRParser.IndirectBrTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#resumeTerm.
    def enterResumeTerm(self, ctx:LLVMIRParser.ResumeTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#resumeTerm.
    def exitResumeTerm(self, ctx:LLVMIRParser.ResumeTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#catchRetTerm.
    def enterCatchRetTerm(self, ctx:LLVMIRParser.CatchRetTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#catchRetTerm.
    def exitCatchRetTerm(self, ctx:LLVMIRParser.CatchRetTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#cleanupRetTerm.
    def enterCleanupRetTerm(self, ctx:LLVMIRParser.CleanupRetTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#cleanupRetTerm.
    def exitCleanupRetTerm(self, ctx:LLVMIRParser.CleanupRetTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#unreachableTerm.
    def enterUnreachableTerm(self, ctx:LLVMIRParser.UnreachableTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#unreachableTerm.
    def exitUnreachableTerm(self, ctx:LLVMIRParser.UnreachableTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#invokeTerm.
    def enterInvokeTerm(self, ctx:LLVMIRParser.InvokeTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#invokeTerm.
    def exitInvokeTerm(self, ctx:LLVMIRParser.InvokeTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#callBrTerm.
    def enterCallBrTerm(self, ctx:LLVMIRParser.CallBrTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#callBrTerm.
    def exitCallBrTerm(self, ctx:LLVMIRParser.CallBrTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#catchSwitchTerm.
    def enterCatchSwitchTerm(self, ctx:LLVMIRParser.CatchSwitchTermContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#catchSwitchTerm.
    def exitCatchSwitchTerm(self, ctx:LLVMIRParser.CatchSwitchTermContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#label.
    def enterLabel(self, ctx:LLVMIRParser.LabelContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#label.
    def exitLabel(self, ctx:LLVMIRParser.LabelContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#case.
    def enterCase(self, ctx:LLVMIRParser.CaseContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#case.
    def exitCase(self, ctx:LLVMIRParser.CaseContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#unwindTarget.
    def enterUnwindTarget(self, ctx:LLVMIRParser.UnwindTargetContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#unwindTarget.
    def exitUnwindTarget(self, ctx:LLVMIRParser.UnwindTargetContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#handlers.
    def enterHandlers(self, ctx:LLVMIRParser.HandlersContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#handlers.
    def exitHandlers(self, ctx:LLVMIRParser.HandlersContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#metadataNode.
    def enterMetadataNode(self, ctx:LLVMIRParser.MetadataNodeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#metadataNode.
    def exitMetadataNode(self, ctx:LLVMIRParser.MetadataNodeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diExpression.
    def enterDiExpression(self, ctx:LLVMIRParser.DiExpressionContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diExpression.
    def exitDiExpression(self, ctx:LLVMIRParser.DiExpressionContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diExpressionField.
    def enterDiExpressionField(self, ctx:LLVMIRParser.DiExpressionFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diExpressionField.
    def exitDiExpressionField(self, ctx:LLVMIRParser.DiExpressionFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#globalField.
    def enterGlobalField(self, ctx:LLVMIRParser.GlobalFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#globalField.
    def exitGlobalField(self, ctx:LLVMIRParser.GlobalFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#section.
    def enterSection(self, ctx:LLVMIRParser.SectionContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#section.
    def exitSection(self, ctx:LLVMIRParser.SectionContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#comdat.
    def enterComdat(self, ctx:LLVMIRParser.ComdatContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#comdat.
    def exitComdat(self, ctx:LLVMIRParser.ComdatContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#partition.
    def enterPartition(self, ctx:LLVMIRParser.PartitionContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#partition.
    def exitPartition(self, ctx:LLVMIRParser.PartitionContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#constant.
    def enterConstant(self, ctx:LLVMIRParser.ConstantContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#constant.
    def exitConstant(self, ctx:LLVMIRParser.ConstantContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#boolConst.
    def enterBoolConst(self, ctx:LLVMIRParser.BoolConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#boolConst.
    def exitBoolConst(self, ctx:LLVMIRParser.BoolConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#intConst.
    def enterIntConst(self, ctx:LLVMIRParser.IntConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#intConst.
    def exitIntConst(self, ctx:LLVMIRParser.IntConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#floatConst.
    def enterFloatConst(self, ctx:LLVMIRParser.FloatConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#floatConst.
    def exitFloatConst(self, ctx:LLVMIRParser.FloatConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#nullConst.
    def enterNullConst(self, ctx:LLVMIRParser.NullConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#nullConst.
    def exitNullConst(self, ctx:LLVMIRParser.NullConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#noneConst.
    def enterNoneConst(self, ctx:LLVMIRParser.NoneConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#noneConst.
    def exitNoneConst(self, ctx:LLVMIRParser.NoneConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#structConst.
    def enterStructConst(self, ctx:LLVMIRParser.StructConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#structConst.
    def exitStructConst(self, ctx:LLVMIRParser.StructConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#arrayConst.
    def enterArrayConst(self, ctx:LLVMIRParser.ArrayConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#arrayConst.
    def exitArrayConst(self, ctx:LLVMIRParser.ArrayConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#vectorConst.
    def enterVectorConst(self, ctx:LLVMIRParser.VectorConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#vectorConst.
    def exitVectorConst(self, ctx:LLVMIRParser.VectorConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#zeroInitializerConst.
    def enterZeroInitializerConst(self, ctx:LLVMIRParser.ZeroInitializerConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#zeroInitializerConst.
    def exitZeroInitializerConst(self, ctx:LLVMIRParser.ZeroInitializerConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#undefConst.
    def enterUndefConst(self, ctx:LLVMIRParser.UndefConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#undefConst.
    def exitUndefConst(self, ctx:LLVMIRParser.UndefConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#poisonConst.
    def enterPoisonConst(self, ctx:LLVMIRParser.PoisonConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#poisonConst.
    def exitPoisonConst(self, ctx:LLVMIRParser.PoisonConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#blockAddressConst.
    def enterBlockAddressConst(self, ctx:LLVMIRParser.BlockAddressConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#blockAddressConst.
    def exitBlockAddressConst(self, ctx:LLVMIRParser.BlockAddressConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#dsoLocalEquivalentConst.
    def enterDsoLocalEquivalentConst(self, ctx:LLVMIRParser.DsoLocalEquivalentConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#dsoLocalEquivalentConst.
    def exitDsoLocalEquivalentConst(self, ctx:LLVMIRParser.DsoLocalEquivalentConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#noCFIConst.
    def enterNoCFIConst(self, ctx:LLVMIRParser.NoCFIConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#noCFIConst.
    def exitNoCFIConst(self, ctx:LLVMIRParser.NoCFIConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#constantExpr.
    def enterConstantExpr(self, ctx:LLVMIRParser.ConstantExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#constantExpr.
    def exitConstantExpr(self, ctx:LLVMIRParser.ConstantExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#typeConst.
    def enterTypeConst(self, ctx:LLVMIRParser.TypeConstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#typeConst.
    def exitTypeConst(self, ctx:LLVMIRParser.TypeConstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#metadataAttachment.
    def enterMetadataAttachment(self, ctx:LLVMIRParser.MetadataAttachmentContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#metadataAttachment.
    def exitMetadataAttachment(self, ctx:LLVMIRParser.MetadataAttachmentContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#mdNode.
    def enterMdNode(self, ctx:LLVMIRParser.MdNodeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#mdNode.
    def exitMdNode(self, ctx:LLVMIRParser.MdNodeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#mdTuple.
    def enterMdTuple(self, ctx:LLVMIRParser.MdTupleContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#mdTuple.
    def exitMdTuple(self, ctx:LLVMIRParser.MdTupleContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#metadata.
    def enterMetadata(self, ctx:LLVMIRParser.MetadataContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#metadata.
    def exitMetadata(self, ctx:LLVMIRParser.MetadataContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diArgList.
    def enterDiArgList(self, ctx:LLVMIRParser.DiArgListContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diArgList.
    def exitDiArgList(self, ctx:LLVMIRParser.DiArgListContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#typeValue.
    def enterTypeValue(self, ctx:LLVMIRParser.TypeValueContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#typeValue.
    def exitTypeValue(self, ctx:LLVMIRParser.TypeValueContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#value.
    def enterValue(self, ctx:LLVMIRParser.ValueContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#value.
    def exitValue(self, ctx:LLVMIRParser.ValueContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#inlineAsm.
    def enterInlineAsm(self, ctx:LLVMIRParser.InlineAsmContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#inlineAsm.
    def exitInlineAsm(self, ctx:LLVMIRParser.InlineAsmContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#mdString.
    def enterMdString(self, ctx:LLVMIRParser.MdStringContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#mdString.
    def exitMdString(self, ctx:LLVMIRParser.MdStringContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#mdFieldOrInt.
    def enterMdFieldOrInt(self, ctx:LLVMIRParser.MdFieldOrIntContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#mdFieldOrInt.
    def exitMdFieldOrInt(self, ctx:LLVMIRParser.MdFieldOrIntContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diSPFlag.
    def enterDiSPFlag(self, ctx:LLVMIRParser.DiSPFlagContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diSPFlag.
    def exitDiSPFlag(self, ctx:LLVMIRParser.DiSPFlagContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#funcAttribute.
    def enterFuncAttribute(self, ctx:LLVMIRParser.FuncAttributeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#funcAttribute.
    def exitFuncAttribute(self, ctx:LLVMIRParser.FuncAttributeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#type.
    def enterType(self, ctx:LLVMIRParser.TypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#type.
    def exitType(self, ctx:LLVMIRParser.TypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#params.
    def enterParams(self, ctx:LLVMIRParser.ParamsContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#params.
    def exitParams(self, ctx:LLVMIRParser.ParamsContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#param.
    def enterParam(self, ctx:LLVMIRParser.ParamContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#param.
    def exitParam(self, ctx:LLVMIRParser.ParamContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#paramAttribute.
    def enterParamAttribute(self, ctx:LLVMIRParser.ParamAttributeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#paramAttribute.
    def exitParamAttribute(self, ctx:LLVMIRParser.ParamAttributeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#attrString.
    def enterAttrString(self, ctx:LLVMIRParser.AttrStringContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#attrString.
    def exitAttrString(self, ctx:LLVMIRParser.AttrStringContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#attrPair.
    def enterAttrPair(self, ctx:LLVMIRParser.AttrPairContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#attrPair.
    def exitAttrPair(self, ctx:LLVMIRParser.AttrPairContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#align.
    def enterAlign(self, ctx:LLVMIRParser.AlignContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#align.
    def exitAlign(self, ctx:LLVMIRParser.AlignContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#alignPair.
    def enterAlignPair(self, ctx:LLVMIRParser.AlignPairContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#alignPair.
    def exitAlignPair(self, ctx:LLVMIRParser.AlignPairContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#alignStack.
    def enterAlignStack(self, ctx:LLVMIRParser.AlignStackContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#alignStack.
    def exitAlignStack(self, ctx:LLVMIRParser.AlignStackContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#alignStackPair.
    def enterAlignStackPair(self, ctx:LLVMIRParser.AlignStackPairContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#alignStackPair.
    def exitAlignStackPair(self, ctx:LLVMIRParser.AlignStackPairContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#allocKind.
    def enterAllocKind(self, ctx:LLVMIRParser.AllocKindContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#allocKind.
    def exitAllocKind(self, ctx:LLVMIRParser.AllocKindContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#allocSize.
    def enterAllocSize(self, ctx:LLVMIRParser.AllocSizeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#allocSize.
    def exitAllocSize(self, ctx:LLVMIRParser.AllocSizeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#unwindTable.
    def enterUnwindTable(self, ctx:LLVMIRParser.UnwindTableContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#unwindTable.
    def exitUnwindTable(self, ctx:LLVMIRParser.UnwindTableContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#vectorScaleRange.
    def enterVectorScaleRange(self, ctx:LLVMIRParser.VectorScaleRangeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#vectorScaleRange.
    def exitVectorScaleRange(self, ctx:LLVMIRParser.VectorScaleRangeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#byRefAttr.
    def enterByRefAttr(self, ctx:LLVMIRParser.ByRefAttrContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#byRefAttr.
    def exitByRefAttr(self, ctx:LLVMIRParser.ByRefAttrContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#byval.
    def enterByval(self, ctx:LLVMIRParser.ByvalContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#byval.
    def exitByval(self, ctx:LLVMIRParser.ByvalContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#dereferenceable.
    def enterDereferenceable(self, ctx:LLVMIRParser.DereferenceableContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#dereferenceable.
    def exitDereferenceable(self, ctx:LLVMIRParser.DereferenceableContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#elementType.
    def enterElementType(self, ctx:LLVMIRParser.ElementTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#elementType.
    def exitElementType(self, ctx:LLVMIRParser.ElementTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#inAlloca.
    def enterInAlloca(self, ctx:LLVMIRParser.InAllocaContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#inAlloca.
    def exitInAlloca(self, ctx:LLVMIRParser.InAllocaContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#paramAttr.
    def enterParamAttr(self, ctx:LLVMIRParser.ParamAttrContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#paramAttr.
    def exitParamAttr(self, ctx:LLVMIRParser.ParamAttrContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#preallocated.
    def enterPreallocated(self, ctx:LLVMIRParser.PreallocatedContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#preallocated.
    def exitPreallocated(self, ctx:LLVMIRParser.PreallocatedContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#structRetAttr.
    def enterStructRetAttr(self, ctx:LLVMIRParser.StructRetAttrContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#structRetAttr.
    def exitStructRetAttr(self, ctx:LLVMIRParser.StructRetAttrContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#firstClassType.
    def enterFirstClassType(self, ctx:LLVMIRParser.FirstClassTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#firstClassType.
    def exitFirstClassType(self, ctx:LLVMIRParser.FirstClassTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#concreteType.
    def enterConcreteType(self, ctx:LLVMIRParser.ConcreteTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#concreteType.
    def exitConcreteType(self, ctx:LLVMIRParser.ConcreteTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#intType.
    def enterIntType(self, ctx:LLVMIRParser.IntTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#intType.
    def exitIntType(self, ctx:LLVMIRParser.IntTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#floatType.
    def enterFloatType(self, ctx:LLVMIRParser.FloatTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#floatType.
    def exitFloatType(self, ctx:LLVMIRParser.FloatTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#pointerType.
    def enterPointerType(self, ctx:LLVMIRParser.PointerTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#pointerType.
    def exitPointerType(self, ctx:LLVMIRParser.PointerTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#vectorType.
    def enterVectorType(self, ctx:LLVMIRParser.VectorTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#vectorType.
    def exitVectorType(self, ctx:LLVMIRParser.VectorTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#labelType.
    def enterLabelType(self, ctx:LLVMIRParser.LabelTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#labelType.
    def exitLabelType(self, ctx:LLVMIRParser.LabelTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#arrayType.
    def enterArrayType(self, ctx:LLVMIRParser.ArrayTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#arrayType.
    def exitArrayType(self, ctx:LLVMIRParser.ArrayTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#structType.
    def enterStructType(self, ctx:LLVMIRParser.StructTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#structType.
    def exitStructType(self, ctx:LLVMIRParser.StructTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#namedType.
    def enterNamedType(self, ctx:LLVMIRParser.NamedTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#namedType.
    def exitNamedType(self, ctx:LLVMIRParser.NamedTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#mmxType.
    def enterMmxType(self, ctx:LLVMIRParser.MmxTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#mmxType.
    def exitMmxType(self, ctx:LLVMIRParser.MmxTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#tokenType.
    def enterTokenType(self, ctx:LLVMIRParser.TokenTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#tokenType.
    def exitTokenType(self, ctx:LLVMIRParser.TokenTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#opaquePointerType.
    def enterOpaquePointerType(self, ctx:LLVMIRParser.OpaquePointerTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#opaquePointerType.
    def exitOpaquePointerType(self, ctx:LLVMIRParser.OpaquePointerTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#addrSpace.
    def enterAddrSpace(self, ctx:LLVMIRParser.AddrSpaceContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#addrSpace.
    def exitAddrSpace(self, ctx:LLVMIRParser.AddrSpaceContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#threadLocal.
    def enterThreadLocal(self, ctx:LLVMIRParser.ThreadLocalContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#threadLocal.
    def exitThreadLocal(self, ctx:LLVMIRParser.ThreadLocalContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#metadataType.
    def enterMetadataType(self, ctx:LLVMIRParser.MetadataTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#metadataType.
    def exitMetadataType(self, ctx:LLVMIRParser.MetadataTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#bitCastExpr.
    def enterBitCastExpr(self, ctx:LLVMIRParser.BitCastExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#bitCastExpr.
    def exitBitCastExpr(self, ctx:LLVMIRParser.BitCastExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#getElementPtrExpr.
    def enterGetElementPtrExpr(self, ctx:LLVMIRParser.GetElementPtrExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#getElementPtrExpr.
    def exitGetElementPtrExpr(self, ctx:LLVMIRParser.GetElementPtrExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#gepIndex.
    def enterGepIndex(self, ctx:LLVMIRParser.GepIndexContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#gepIndex.
    def exitGepIndex(self, ctx:LLVMIRParser.GepIndexContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#addrSpaceCastExpr.
    def enterAddrSpaceCastExpr(self, ctx:LLVMIRParser.AddrSpaceCastExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#addrSpaceCastExpr.
    def exitAddrSpaceCastExpr(self, ctx:LLVMIRParser.AddrSpaceCastExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#intToPtrExpr.
    def enterIntToPtrExpr(self, ctx:LLVMIRParser.IntToPtrExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#intToPtrExpr.
    def exitIntToPtrExpr(self, ctx:LLVMIRParser.IntToPtrExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#iCmpExpr.
    def enterICmpExpr(self, ctx:LLVMIRParser.ICmpExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#iCmpExpr.
    def exitICmpExpr(self, ctx:LLVMIRParser.ICmpExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fCmpExpr.
    def enterFCmpExpr(self, ctx:LLVMIRParser.FCmpExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fCmpExpr.
    def exitFCmpExpr(self, ctx:LLVMIRParser.FCmpExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#selectExpr.
    def enterSelectExpr(self, ctx:LLVMIRParser.SelectExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#selectExpr.
    def exitSelectExpr(self, ctx:LLVMIRParser.SelectExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#truncExpr.
    def enterTruncExpr(self, ctx:LLVMIRParser.TruncExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#truncExpr.
    def exitTruncExpr(self, ctx:LLVMIRParser.TruncExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#zExtExpr.
    def enterZExtExpr(self, ctx:LLVMIRParser.ZExtExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#zExtExpr.
    def exitZExtExpr(self, ctx:LLVMIRParser.ZExtExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sExtExpr.
    def enterSExtExpr(self, ctx:LLVMIRParser.SExtExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sExtExpr.
    def exitSExtExpr(self, ctx:LLVMIRParser.SExtExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fpTruncExpr.
    def enterFpTruncExpr(self, ctx:LLVMIRParser.FpTruncExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fpTruncExpr.
    def exitFpTruncExpr(self, ctx:LLVMIRParser.FpTruncExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fpExtExpr.
    def enterFpExtExpr(self, ctx:LLVMIRParser.FpExtExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fpExtExpr.
    def exitFpExtExpr(self, ctx:LLVMIRParser.FpExtExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fpToUiExpr.
    def enterFpToUiExpr(self, ctx:LLVMIRParser.FpToUiExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fpToUiExpr.
    def exitFpToUiExpr(self, ctx:LLVMIRParser.FpToUiExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fpToSiExpr.
    def enterFpToSiExpr(self, ctx:LLVMIRParser.FpToSiExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fpToSiExpr.
    def exitFpToSiExpr(self, ctx:LLVMIRParser.FpToSiExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#uiToFpExpr.
    def enterUiToFpExpr(self, ctx:LLVMIRParser.UiToFpExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#uiToFpExpr.
    def exitUiToFpExpr(self, ctx:LLVMIRParser.UiToFpExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#siToFpExpr.
    def enterSiToFpExpr(self, ctx:LLVMIRParser.SiToFpExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#siToFpExpr.
    def exitSiToFpExpr(self, ctx:LLVMIRParser.SiToFpExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#ptrToIntExpr.
    def enterPtrToIntExpr(self, ctx:LLVMIRParser.PtrToIntExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#ptrToIntExpr.
    def exitPtrToIntExpr(self, ctx:LLVMIRParser.PtrToIntExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#extractElementExpr.
    def enterExtractElementExpr(self, ctx:LLVMIRParser.ExtractElementExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#extractElementExpr.
    def exitExtractElementExpr(self, ctx:LLVMIRParser.ExtractElementExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#insertElementExpr.
    def enterInsertElementExpr(self, ctx:LLVMIRParser.InsertElementExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#insertElementExpr.
    def exitInsertElementExpr(self, ctx:LLVMIRParser.InsertElementExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#shuffleVectorExpr.
    def enterShuffleVectorExpr(self, ctx:LLVMIRParser.ShuffleVectorExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#shuffleVectorExpr.
    def exitShuffleVectorExpr(self, ctx:LLVMIRParser.ShuffleVectorExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#shlExpr.
    def enterShlExpr(self, ctx:LLVMIRParser.ShlExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#shlExpr.
    def exitShlExpr(self, ctx:LLVMIRParser.ShlExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#lShrExpr.
    def enterLShrExpr(self, ctx:LLVMIRParser.LShrExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#lShrExpr.
    def exitLShrExpr(self, ctx:LLVMIRParser.LShrExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#aShrExpr.
    def enterAShrExpr(self, ctx:LLVMIRParser.AShrExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#aShrExpr.
    def exitAShrExpr(self, ctx:LLVMIRParser.AShrExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#andExpr.
    def enterAndExpr(self, ctx:LLVMIRParser.AndExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#andExpr.
    def exitAndExpr(self, ctx:LLVMIRParser.AndExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#orExpr.
    def enterOrExpr(self, ctx:LLVMIRParser.OrExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#orExpr.
    def exitOrExpr(self, ctx:LLVMIRParser.OrExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#xorExpr.
    def enterXorExpr(self, ctx:LLVMIRParser.XorExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#xorExpr.
    def exitXorExpr(self, ctx:LLVMIRParser.XorExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#addExpr.
    def enterAddExpr(self, ctx:LLVMIRParser.AddExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#addExpr.
    def exitAddExpr(self, ctx:LLVMIRParser.AddExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#subExpr.
    def enterSubExpr(self, ctx:LLVMIRParser.SubExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#subExpr.
    def exitSubExpr(self, ctx:LLVMIRParser.SubExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#mulExpr.
    def enterMulExpr(self, ctx:LLVMIRParser.MulExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#mulExpr.
    def exitMulExpr(self, ctx:LLVMIRParser.MulExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fNegExpr.
    def enterFNegExpr(self, ctx:LLVMIRParser.FNegExprContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fNegExpr.
    def exitFNegExpr(self, ctx:LLVMIRParser.FNegExprContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#localDefInst.
    def enterLocalDefInst(self, ctx:LLVMIRParser.LocalDefInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#localDefInst.
    def exitLocalDefInst(self, ctx:LLVMIRParser.LocalDefInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#valueInstruction.
    def enterValueInstruction(self, ctx:LLVMIRParser.ValueInstructionContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#valueInstruction.
    def exitValueInstruction(self, ctx:LLVMIRParser.ValueInstructionContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#storeInst.
    def enterStoreInst(self, ctx:LLVMIRParser.StoreInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#storeInst.
    def exitStoreInst(self, ctx:LLVMIRParser.StoreInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#syncScope.
    def enterSyncScope(self, ctx:LLVMIRParser.SyncScopeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#syncScope.
    def exitSyncScope(self, ctx:LLVMIRParser.SyncScopeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fenceInst.
    def enterFenceInst(self, ctx:LLVMIRParser.FenceInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fenceInst.
    def exitFenceInst(self, ctx:LLVMIRParser.FenceInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fNegInst.
    def enterFNegInst(self, ctx:LLVMIRParser.FNegInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fNegInst.
    def exitFNegInst(self, ctx:LLVMIRParser.FNegInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#addInst.
    def enterAddInst(self, ctx:LLVMIRParser.AddInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#addInst.
    def exitAddInst(self, ctx:LLVMIRParser.AddInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fAddInst.
    def enterFAddInst(self, ctx:LLVMIRParser.FAddInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fAddInst.
    def exitFAddInst(self, ctx:LLVMIRParser.FAddInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#subInst.
    def enterSubInst(self, ctx:LLVMIRParser.SubInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#subInst.
    def exitSubInst(self, ctx:LLVMIRParser.SubInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fSubInst.
    def enterFSubInst(self, ctx:LLVMIRParser.FSubInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fSubInst.
    def exitFSubInst(self, ctx:LLVMIRParser.FSubInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#mulInst.
    def enterMulInst(self, ctx:LLVMIRParser.MulInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#mulInst.
    def exitMulInst(self, ctx:LLVMIRParser.MulInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fMulInst.
    def enterFMulInst(self, ctx:LLVMIRParser.FMulInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fMulInst.
    def exitFMulInst(self, ctx:LLVMIRParser.FMulInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#uDivInst.
    def enterUDivInst(self, ctx:LLVMIRParser.UDivInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#uDivInst.
    def exitUDivInst(self, ctx:LLVMIRParser.UDivInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sDivInst.
    def enterSDivInst(self, ctx:LLVMIRParser.SDivInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sDivInst.
    def exitSDivInst(self, ctx:LLVMIRParser.SDivInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fDivInst.
    def enterFDivInst(self, ctx:LLVMIRParser.FDivInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fDivInst.
    def exitFDivInst(self, ctx:LLVMIRParser.FDivInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#uRemInst.
    def enterURemInst(self, ctx:LLVMIRParser.URemInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#uRemInst.
    def exitURemInst(self, ctx:LLVMIRParser.URemInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sRemInst.
    def enterSRemInst(self, ctx:LLVMIRParser.SRemInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sRemInst.
    def exitSRemInst(self, ctx:LLVMIRParser.SRemInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fRemInst.
    def enterFRemInst(self, ctx:LLVMIRParser.FRemInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fRemInst.
    def exitFRemInst(self, ctx:LLVMIRParser.FRemInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#shlInst.
    def enterShlInst(self, ctx:LLVMIRParser.ShlInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#shlInst.
    def exitShlInst(self, ctx:LLVMIRParser.ShlInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#lShrInst.
    def enterLShrInst(self, ctx:LLVMIRParser.LShrInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#lShrInst.
    def exitLShrInst(self, ctx:LLVMIRParser.LShrInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#aShrInst.
    def enterAShrInst(self, ctx:LLVMIRParser.AShrInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#aShrInst.
    def exitAShrInst(self, ctx:LLVMIRParser.AShrInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#andInst.
    def enterAndInst(self, ctx:LLVMIRParser.AndInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#andInst.
    def exitAndInst(self, ctx:LLVMIRParser.AndInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#orInst.
    def enterOrInst(self, ctx:LLVMIRParser.OrInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#orInst.
    def exitOrInst(self, ctx:LLVMIRParser.OrInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#xorInst.
    def enterXorInst(self, ctx:LLVMIRParser.XorInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#xorInst.
    def exitXorInst(self, ctx:LLVMIRParser.XorInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#extractElementInst.
    def enterExtractElementInst(self, ctx:LLVMIRParser.ExtractElementInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#extractElementInst.
    def exitExtractElementInst(self, ctx:LLVMIRParser.ExtractElementInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#insertElementInst.
    def enterInsertElementInst(self, ctx:LLVMIRParser.InsertElementInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#insertElementInst.
    def exitInsertElementInst(self, ctx:LLVMIRParser.InsertElementInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#shuffleVectorInst.
    def enterShuffleVectorInst(self, ctx:LLVMIRParser.ShuffleVectorInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#shuffleVectorInst.
    def exitShuffleVectorInst(self, ctx:LLVMIRParser.ShuffleVectorInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#extractValueInst.
    def enterExtractValueInst(self, ctx:LLVMIRParser.ExtractValueInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#extractValueInst.
    def exitExtractValueInst(self, ctx:LLVMIRParser.ExtractValueInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#insertValueInst.
    def enterInsertValueInst(self, ctx:LLVMIRParser.InsertValueInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#insertValueInst.
    def exitInsertValueInst(self, ctx:LLVMIRParser.InsertValueInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#allocaInst.
    def enterAllocaInst(self, ctx:LLVMIRParser.AllocaInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#allocaInst.
    def exitAllocaInst(self, ctx:LLVMIRParser.AllocaInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#loadInst.
    def enterLoadInst(self, ctx:LLVMIRParser.LoadInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#loadInst.
    def exitLoadInst(self, ctx:LLVMIRParser.LoadInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#cmpXchgInst.
    def enterCmpXchgInst(self, ctx:LLVMIRParser.CmpXchgInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#cmpXchgInst.
    def exitCmpXchgInst(self, ctx:LLVMIRParser.CmpXchgInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#atomicRMWInst.
    def enterAtomicRMWInst(self, ctx:LLVMIRParser.AtomicRMWInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#atomicRMWInst.
    def exitAtomicRMWInst(self, ctx:LLVMIRParser.AtomicRMWInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#getElementPtrInst.
    def enterGetElementPtrInst(self, ctx:LLVMIRParser.GetElementPtrInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#getElementPtrInst.
    def exitGetElementPtrInst(self, ctx:LLVMIRParser.GetElementPtrInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#truncInst.
    def enterTruncInst(self, ctx:LLVMIRParser.TruncInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#truncInst.
    def exitTruncInst(self, ctx:LLVMIRParser.TruncInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#zExtInst.
    def enterZExtInst(self, ctx:LLVMIRParser.ZExtInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#zExtInst.
    def exitZExtInst(self, ctx:LLVMIRParser.ZExtInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sExtInst.
    def enterSExtInst(self, ctx:LLVMIRParser.SExtInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sExtInst.
    def exitSExtInst(self, ctx:LLVMIRParser.SExtInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fpTruncInst.
    def enterFpTruncInst(self, ctx:LLVMIRParser.FpTruncInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fpTruncInst.
    def exitFpTruncInst(self, ctx:LLVMIRParser.FpTruncInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fpExtInst.
    def enterFpExtInst(self, ctx:LLVMIRParser.FpExtInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fpExtInst.
    def exitFpExtInst(self, ctx:LLVMIRParser.FpExtInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fpToUiInst.
    def enterFpToUiInst(self, ctx:LLVMIRParser.FpToUiInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fpToUiInst.
    def exitFpToUiInst(self, ctx:LLVMIRParser.FpToUiInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fpToSiInst.
    def enterFpToSiInst(self, ctx:LLVMIRParser.FpToSiInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fpToSiInst.
    def exitFpToSiInst(self, ctx:LLVMIRParser.FpToSiInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#uiToFpInst.
    def enterUiToFpInst(self, ctx:LLVMIRParser.UiToFpInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#uiToFpInst.
    def exitUiToFpInst(self, ctx:LLVMIRParser.UiToFpInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#siToFpInst.
    def enterSiToFpInst(self, ctx:LLVMIRParser.SiToFpInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#siToFpInst.
    def exitSiToFpInst(self, ctx:LLVMIRParser.SiToFpInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#ptrToIntInst.
    def enterPtrToIntInst(self, ctx:LLVMIRParser.PtrToIntInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#ptrToIntInst.
    def exitPtrToIntInst(self, ctx:LLVMIRParser.PtrToIntInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#intToPtrInst.
    def enterIntToPtrInst(self, ctx:LLVMIRParser.IntToPtrInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#intToPtrInst.
    def exitIntToPtrInst(self, ctx:LLVMIRParser.IntToPtrInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#bitCastInst.
    def enterBitCastInst(self, ctx:LLVMIRParser.BitCastInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#bitCastInst.
    def exitBitCastInst(self, ctx:LLVMIRParser.BitCastInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#addrSpaceCastInst.
    def enterAddrSpaceCastInst(self, ctx:LLVMIRParser.AddrSpaceCastInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#addrSpaceCastInst.
    def exitAddrSpaceCastInst(self, ctx:LLVMIRParser.AddrSpaceCastInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#iCmpInst.
    def enterICmpInst(self, ctx:LLVMIRParser.ICmpInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#iCmpInst.
    def exitICmpInst(self, ctx:LLVMIRParser.ICmpInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fCmpInst.
    def enterFCmpInst(self, ctx:LLVMIRParser.FCmpInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fCmpInst.
    def exitFCmpInst(self, ctx:LLVMIRParser.FCmpInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#phiInst.
    def enterPhiInst(self, ctx:LLVMIRParser.PhiInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#phiInst.
    def exitPhiInst(self, ctx:LLVMIRParser.PhiInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#selectInst.
    def enterSelectInst(self, ctx:LLVMIRParser.SelectInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#selectInst.
    def exitSelectInst(self, ctx:LLVMIRParser.SelectInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#freezeInst.
    def enterFreezeInst(self, ctx:LLVMIRParser.FreezeInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#freezeInst.
    def exitFreezeInst(self, ctx:LLVMIRParser.FreezeInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#callInst.
    def enterCallInst(self, ctx:LLVMIRParser.CallInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#callInst.
    def exitCallInst(self, ctx:LLVMIRParser.CallInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#vaargInst.
    def enterVaargInst(self, ctx:LLVMIRParser.VaargInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#vaargInst.
    def exitVaargInst(self, ctx:LLVMIRParser.VaargInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#landingPadInst.
    def enterLandingPadInst(self, ctx:LLVMIRParser.LandingPadInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#landingPadInst.
    def exitLandingPadInst(self, ctx:LLVMIRParser.LandingPadInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#catchPadInst.
    def enterCatchPadInst(self, ctx:LLVMIRParser.CatchPadInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#catchPadInst.
    def exitCatchPadInst(self, ctx:LLVMIRParser.CatchPadInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#cleanupPadInst.
    def enterCleanupPadInst(self, ctx:LLVMIRParser.CleanupPadInstContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#cleanupPadInst.
    def exitCleanupPadInst(self, ctx:LLVMIRParser.CleanupPadInstContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#inc.
    def enterInc(self, ctx:LLVMIRParser.IncContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#inc.
    def exitInc(self, ctx:LLVMIRParser.IncContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#operandBundle.
    def enterOperandBundle(self, ctx:LLVMIRParser.OperandBundleContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#operandBundle.
    def exitOperandBundle(self, ctx:LLVMIRParser.OperandBundleContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#clause.
    def enterClause(self, ctx:LLVMIRParser.ClauseContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#clause.
    def exitClause(self, ctx:LLVMIRParser.ClauseContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#args.
    def enterArgs(self, ctx:LLVMIRParser.ArgsContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#args.
    def exitArgs(self, ctx:LLVMIRParser.ArgsContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#arg.
    def enterArg(self, ctx:LLVMIRParser.ArgContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#arg.
    def exitArg(self, ctx:LLVMIRParser.ArgContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#exceptionArg.
    def enterExceptionArg(self, ctx:LLVMIRParser.ExceptionArgContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#exceptionArg.
    def exitExceptionArg(self, ctx:LLVMIRParser.ExceptionArgContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#exceptionPad.
    def enterExceptionPad(self, ctx:LLVMIRParser.ExceptionPadContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#exceptionPad.
    def exitExceptionPad(self, ctx:LLVMIRParser.ExceptionPadContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#externalLinkage.
    def enterExternalLinkage(self, ctx:LLVMIRParser.ExternalLinkageContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#externalLinkage.
    def exitExternalLinkage(self, ctx:LLVMIRParser.ExternalLinkageContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#internalLinkage.
    def enterInternalLinkage(self, ctx:LLVMIRParser.InternalLinkageContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#internalLinkage.
    def exitInternalLinkage(self, ctx:LLVMIRParser.InternalLinkageContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#linkage.
    def enterLinkage(self, ctx:LLVMIRParser.LinkageContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#linkage.
    def exitLinkage(self, ctx:LLVMIRParser.LinkageContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#preemption.
    def enterPreemption(self, ctx:LLVMIRParser.PreemptionContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#preemption.
    def exitPreemption(self, ctx:LLVMIRParser.PreemptionContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#visibility.
    def enterVisibility(self, ctx:LLVMIRParser.VisibilityContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#visibility.
    def exitVisibility(self, ctx:LLVMIRParser.VisibilityContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#dllStorageClass.
    def enterDllStorageClass(self, ctx:LLVMIRParser.DllStorageClassContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#dllStorageClass.
    def exitDllStorageClass(self, ctx:LLVMIRParser.DllStorageClassContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#tlsModel.
    def enterTlsModel(self, ctx:LLVMIRParser.TlsModelContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#tlsModel.
    def exitTlsModel(self, ctx:LLVMIRParser.TlsModelContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#unnamedAddr.
    def enterUnnamedAddr(self, ctx:LLVMIRParser.UnnamedAddrContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#unnamedAddr.
    def exitUnnamedAddr(self, ctx:LLVMIRParser.UnnamedAddrContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#externallyInitialized.
    def enterExternallyInitialized(self, ctx:LLVMIRParser.ExternallyInitializedContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#externallyInitialized.
    def exitExternallyInitialized(self, ctx:LLVMIRParser.ExternallyInitializedContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#immutable.
    def enterImmutable(self, ctx:LLVMIRParser.ImmutableContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#immutable.
    def exitImmutable(self, ctx:LLVMIRParser.ImmutableContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#funcAttr.
    def enterFuncAttr(self, ctx:LLVMIRParser.FuncAttrContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#funcAttr.
    def exitFuncAttr(self, ctx:LLVMIRParser.FuncAttrContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#distinct.
    def enterDistinct(self, ctx:LLVMIRParser.DistinctContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#distinct.
    def exitDistinct(self, ctx:LLVMIRParser.DistinctContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#inBounds.
    def enterInBounds(self, ctx:LLVMIRParser.InBoundsContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#inBounds.
    def exitInBounds(self, ctx:LLVMIRParser.InBoundsContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#returnAttr.
    def enterReturnAttr(self, ctx:LLVMIRParser.ReturnAttrContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#returnAttr.
    def exitReturnAttr(self, ctx:LLVMIRParser.ReturnAttrContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#overflowFlag.
    def enterOverflowFlag(self, ctx:LLVMIRParser.OverflowFlagContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#overflowFlag.
    def exitOverflowFlag(self, ctx:LLVMIRParser.OverflowFlagContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#iPred.
    def enterIPred(self, ctx:LLVMIRParser.IPredContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#iPred.
    def exitIPred(self, ctx:LLVMIRParser.IPredContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fPred.
    def enterFPred(self, ctx:LLVMIRParser.FPredContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fPred.
    def exitFPred(self, ctx:LLVMIRParser.FPredContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#atomicOrdering.
    def enterAtomicOrdering(self, ctx:LLVMIRParser.AtomicOrderingContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#atomicOrdering.
    def exitAtomicOrdering(self, ctx:LLVMIRParser.AtomicOrderingContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#callingConvEnum.
    def enterCallingConvEnum(self, ctx:LLVMIRParser.CallingConvEnumContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#callingConvEnum.
    def exitCallingConvEnum(self, ctx:LLVMIRParser.CallingConvEnumContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fastMathFlag.
    def enterFastMathFlag(self, ctx:LLVMIRParser.FastMathFlagContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fastMathFlag.
    def exitFastMathFlag(self, ctx:LLVMIRParser.FastMathFlagContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#atomicOp.
    def enterAtomicOp(self, ctx:LLVMIRParser.AtomicOpContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#atomicOp.
    def exitAtomicOp(self, ctx:LLVMIRParser.AtomicOpContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#floatKind.
    def enterFloatKind(self, ctx:LLVMIRParser.FloatKindContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#floatKind.
    def exitFloatKind(self, ctx:LLVMIRParser.FloatKindContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#specializedMDNode.
    def enterSpecializedMDNode(self, ctx:LLVMIRParser.SpecializedMDNodeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#specializedMDNode.
    def exitSpecializedMDNode(self, ctx:LLVMIRParser.SpecializedMDNodeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diBasicType.
    def enterDiBasicType(self, ctx:LLVMIRParser.DiBasicTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diBasicType.
    def exitDiBasicType(self, ctx:LLVMIRParser.DiBasicTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diCommonBlock.
    def enterDiCommonBlock(self, ctx:LLVMIRParser.DiCommonBlockContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diCommonBlock.
    def exitDiCommonBlock(self, ctx:LLVMIRParser.DiCommonBlockContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diCompileUnit.
    def enterDiCompileUnit(self, ctx:LLVMIRParser.DiCompileUnitContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diCompileUnit.
    def exitDiCompileUnit(self, ctx:LLVMIRParser.DiCompileUnitContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diCompositeType.
    def enterDiCompositeType(self, ctx:LLVMIRParser.DiCompositeTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diCompositeType.
    def exitDiCompositeType(self, ctx:LLVMIRParser.DiCompositeTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diCompositeTypeField.
    def enterDiCompositeTypeField(self, ctx:LLVMIRParser.DiCompositeTypeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diCompositeTypeField.
    def exitDiCompositeTypeField(self, ctx:LLVMIRParser.DiCompositeTypeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diDerivedType.
    def enterDiDerivedType(self, ctx:LLVMIRParser.DiDerivedTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diDerivedType.
    def exitDiDerivedType(self, ctx:LLVMIRParser.DiDerivedTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diDerivedTypeField.
    def enterDiDerivedTypeField(self, ctx:LLVMIRParser.DiDerivedTypeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diDerivedTypeField.
    def exitDiDerivedTypeField(self, ctx:LLVMIRParser.DiDerivedTypeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diEnumerator.
    def enterDiEnumerator(self, ctx:LLVMIRParser.DiEnumeratorContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diEnumerator.
    def exitDiEnumerator(self, ctx:LLVMIRParser.DiEnumeratorContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diEnumeratorField.
    def enterDiEnumeratorField(self, ctx:LLVMIRParser.DiEnumeratorFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diEnumeratorField.
    def exitDiEnumeratorField(self, ctx:LLVMIRParser.DiEnumeratorFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diFile.
    def enterDiFile(self, ctx:LLVMIRParser.DiFileContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diFile.
    def exitDiFile(self, ctx:LLVMIRParser.DiFileContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diFileField.
    def enterDiFileField(self, ctx:LLVMIRParser.DiFileFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diFileField.
    def exitDiFileField(self, ctx:LLVMIRParser.DiFileFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diGlobalVariable.
    def enterDiGlobalVariable(self, ctx:LLVMIRParser.DiGlobalVariableContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diGlobalVariable.
    def exitDiGlobalVariable(self, ctx:LLVMIRParser.DiGlobalVariableContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diGlobalVariableField.
    def enterDiGlobalVariableField(self, ctx:LLVMIRParser.DiGlobalVariableFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diGlobalVariableField.
    def exitDiGlobalVariableField(self, ctx:LLVMIRParser.DiGlobalVariableFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diGlobalVariableExpression.
    def enterDiGlobalVariableExpression(self, ctx:LLVMIRParser.DiGlobalVariableExpressionContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diGlobalVariableExpression.
    def exitDiGlobalVariableExpression(self, ctx:LLVMIRParser.DiGlobalVariableExpressionContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diGlobalVariableExpressionField.
    def enterDiGlobalVariableExpressionField(self, ctx:LLVMIRParser.DiGlobalVariableExpressionFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diGlobalVariableExpressionField.
    def exitDiGlobalVariableExpressionField(self, ctx:LLVMIRParser.DiGlobalVariableExpressionFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diImportedEntity.
    def enterDiImportedEntity(self, ctx:LLVMIRParser.DiImportedEntityContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diImportedEntity.
    def exitDiImportedEntity(self, ctx:LLVMIRParser.DiImportedEntityContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diImportedEntityField.
    def enterDiImportedEntityField(self, ctx:LLVMIRParser.DiImportedEntityFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diImportedEntityField.
    def exitDiImportedEntityField(self, ctx:LLVMIRParser.DiImportedEntityFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLabel.
    def enterDiLabel(self, ctx:LLVMIRParser.DiLabelContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLabel.
    def exitDiLabel(self, ctx:LLVMIRParser.DiLabelContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLabelField.
    def enterDiLabelField(self, ctx:LLVMIRParser.DiLabelFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLabelField.
    def exitDiLabelField(self, ctx:LLVMIRParser.DiLabelFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLexicalBlock.
    def enterDiLexicalBlock(self, ctx:LLVMIRParser.DiLexicalBlockContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLexicalBlock.
    def exitDiLexicalBlock(self, ctx:LLVMIRParser.DiLexicalBlockContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLexicalBlockField.
    def enterDiLexicalBlockField(self, ctx:LLVMIRParser.DiLexicalBlockFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLexicalBlockField.
    def exitDiLexicalBlockField(self, ctx:LLVMIRParser.DiLexicalBlockFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLexicalBlockFile.
    def enterDiLexicalBlockFile(self, ctx:LLVMIRParser.DiLexicalBlockFileContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLexicalBlockFile.
    def exitDiLexicalBlockFile(self, ctx:LLVMIRParser.DiLexicalBlockFileContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLexicalBlockFileField.
    def enterDiLexicalBlockFileField(self, ctx:LLVMIRParser.DiLexicalBlockFileFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLexicalBlockFileField.
    def exitDiLexicalBlockFileField(self, ctx:LLVMIRParser.DiLexicalBlockFileFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLocalVariable.
    def enterDiLocalVariable(self, ctx:LLVMIRParser.DiLocalVariableContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLocalVariable.
    def exitDiLocalVariable(self, ctx:LLVMIRParser.DiLocalVariableContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLocalVariableField.
    def enterDiLocalVariableField(self, ctx:LLVMIRParser.DiLocalVariableFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLocalVariableField.
    def exitDiLocalVariableField(self, ctx:LLVMIRParser.DiLocalVariableFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLocation.
    def enterDiLocation(self, ctx:LLVMIRParser.DiLocationContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLocation.
    def exitDiLocation(self, ctx:LLVMIRParser.DiLocationContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diLocationField.
    def enterDiLocationField(self, ctx:LLVMIRParser.DiLocationFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diLocationField.
    def exitDiLocationField(self, ctx:LLVMIRParser.DiLocationFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diMacro.
    def enterDiMacro(self, ctx:LLVMIRParser.DiMacroContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diMacro.
    def exitDiMacro(self, ctx:LLVMIRParser.DiMacroContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diMacroField.
    def enterDiMacroField(self, ctx:LLVMIRParser.DiMacroFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diMacroField.
    def exitDiMacroField(self, ctx:LLVMIRParser.DiMacroFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diMacroFile.
    def enterDiMacroFile(self, ctx:LLVMIRParser.DiMacroFileContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diMacroFile.
    def exitDiMacroFile(self, ctx:LLVMIRParser.DiMacroFileContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diMacroFileField.
    def enterDiMacroFileField(self, ctx:LLVMIRParser.DiMacroFileFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diMacroFileField.
    def exitDiMacroFileField(self, ctx:LLVMIRParser.DiMacroFileFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diModule.
    def enterDiModule(self, ctx:LLVMIRParser.DiModuleContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diModule.
    def exitDiModule(self, ctx:LLVMIRParser.DiModuleContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diModuleField.
    def enterDiModuleField(self, ctx:LLVMIRParser.DiModuleFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diModuleField.
    def exitDiModuleField(self, ctx:LLVMIRParser.DiModuleFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diNamespace.
    def enterDiNamespace(self, ctx:LLVMIRParser.DiNamespaceContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diNamespace.
    def exitDiNamespace(self, ctx:LLVMIRParser.DiNamespaceContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diNamespaceField.
    def enterDiNamespaceField(self, ctx:LLVMIRParser.DiNamespaceFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diNamespaceField.
    def exitDiNamespaceField(self, ctx:LLVMIRParser.DiNamespaceFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diObjCProperty.
    def enterDiObjCProperty(self, ctx:LLVMIRParser.DiObjCPropertyContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diObjCProperty.
    def exitDiObjCProperty(self, ctx:LLVMIRParser.DiObjCPropertyContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diObjCPropertyField.
    def enterDiObjCPropertyField(self, ctx:LLVMIRParser.DiObjCPropertyFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diObjCPropertyField.
    def exitDiObjCPropertyField(self, ctx:LLVMIRParser.DiObjCPropertyFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diStringType.
    def enterDiStringType(self, ctx:LLVMIRParser.DiStringTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diStringType.
    def exitDiStringType(self, ctx:LLVMIRParser.DiStringTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diStringTypeField.
    def enterDiStringTypeField(self, ctx:LLVMIRParser.DiStringTypeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diStringTypeField.
    def exitDiStringTypeField(self, ctx:LLVMIRParser.DiStringTypeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diSubprogram.
    def enterDiSubprogram(self, ctx:LLVMIRParser.DiSubprogramContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diSubprogram.
    def exitDiSubprogram(self, ctx:LLVMIRParser.DiSubprogramContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diSubprogramField.
    def enterDiSubprogramField(self, ctx:LLVMIRParser.DiSubprogramFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diSubprogramField.
    def exitDiSubprogramField(self, ctx:LLVMIRParser.DiSubprogramFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diSubrange.
    def enterDiSubrange(self, ctx:LLVMIRParser.DiSubrangeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diSubrange.
    def exitDiSubrange(self, ctx:LLVMIRParser.DiSubrangeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diSubrangeField.
    def enterDiSubrangeField(self, ctx:LLVMIRParser.DiSubrangeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diSubrangeField.
    def exitDiSubrangeField(self, ctx:LLVMIRParser.DiSubrangeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diSubroutineType.
    def enterDiSubroutineType(self, ctx:LLVMIRParser.DiSubroutineTypeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diSubroutineType.
    def exitDiSubroutineType(self, ctx:LLVMIRParser.DiSubroutineTypeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diTemplateTypeParameter.
    def enterDiTemplateTypeParameter(self, ctx:LLVMIRParser.DiTemplateTypeParameterContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diTemplateTypeParameter.
    def exitDiTemplateTypeParameter(self, ctx:LLVMIRParser.DiTemplateTypeParameterContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diTemplateValueParameter.
    def enterDiTemplateValueParameter(self, ctx:LLVMIRParser.DiTemplateValueParameterContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diTemplateValueParameter.
    def exitDiTemplateValueParameter(self, ctx:LLVMIRParser.DiTemplateValueParameterContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#genericDiNode.
    def enterGenericDiNode(self, ctx:LLVMIRParser.GenericDiNodeContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#genericDiNode.
    def exitGenericDiNode(self, ctx:LLVMIRParser.GenericDiNodeContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diTemplateTypeParameterField.
    def enterDiTemplateTypeParameterField(self, ctx:LLVMIRParser.DiTemplateTypeParameterFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diTemplateTypeParameterField.
    def exitDiTemplateTypeParameterField(self, ctx:LLVMIRParser.DiTemplateTypeParameterFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diCompileUnitField.
    def enterDiCompileUnitField(self, ctx:LLVMIRParser.DiCompileUnitFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diCompileUnitField.
    def exitDiCompileUnitField(self, ctx:LLVMIRParser.DiCompileUnitFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diCommonBlockField.
    def enterDiCommonBlockField(self, ctx:LLVMIRParser.DiCommonBlockFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diCommonBlockField.
    def exitDiCommonBlockField(self, ctx:LLVMIRParser.DiCommonBlockFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diBasicTypeField.
    def enterDiBasicTypeField(self, ctx:LLVMIRParser.DiBasicTypeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diBasicTypeField.
    def exitDiBasicTypeField(self, ctx:LLVMIRParser.DiBasicTypeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#genericDINodeField.
    def enterGenericDINodeField(self, ctx:LLVMIRParser.GenericDINodeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#genericDINodeField.
    def exitGenericDINodeField(self, ctx:LLVMIRParser.GenericDINodeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#tagField.
    def enterTagField(self, ctx:LLVMIRParser.TagFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#tagField.
    def exitTagField(self, ctx:LLVMIRParser.TagFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#headerField.
    def enterHeaderField(self, ctx:LLVMIRParser.HeaderFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#headerField.
    def exitHeaderField(self, ctx:LLVMIRParser.HeaderFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#operandsField.
    def enterOperandsField(self, ctx:LLVMIRParser.OperandsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#operandsField.
    def exitOperandsField(self, ctx:LLVMIRParser.OperandsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diTemplateValueParameterField.
    def enterDiTemplateValueParameterField(self, ctx:LLVMIRParser.DiTemplateValueParameterFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diTemplateValueParameterField.
    def exitDiTemplateValueParameterField(self, ctx:LLVMIRParser.DiTemplateValueParameterFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#nameField.
    def enterNameField(self, ctx:LLVMIRParser.NameFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#nameField.
    def exitNameField(self, ctx:LLVMIRParser.NameFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#typeField.
    def enterTypeField(self, ctx:LLVMIRParser.TypeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#typeField.
    def exitTypeField(self, ctx:LLVMIRParser.TypeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#defaultedField.
    def enterDefaultedField(self, ctx:LLVMIRParser.DefaultedFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#defaultedField.
    def exitDefaultedField(self, ctx:LLVMIRParser.DefaultedFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#valueField.
    def enterValueField(self, ctx:LLVMIRParser.ValueFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#valueField.
    def exitValueField(self, ctx:LLVMIRParser.ValueFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#mdField.
    def enterMdField(self, ctx:LLVMIRParser.MdFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#mdField.
    def exitMdField(self, ctx:LLVMIRParser.MdFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diSubroutineTypeField.
    def enterDiSubroutineTypeField(self, ctx:LLVMIRParser.DiSubroutineTypeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diSubroutineTypeField.
    def exitDiSubroutineTypeField(self, ctx:LLVMIRParser.DiSubroutineTypeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#flagsField.
    def enterFlagsField(self, ctx:LLVMIRParser.FlagsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#flagsField.
    def exitFlagsField(self, ctx:LLVMIRParser.FlagsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#diFlags.
    def enterDiFlags(self, ctx:LLVMIRParser.DiFlagsContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#diFlags.
    def exitDiFlags(self, ctx:LLVMIRParser.DiFlagsContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#ccField.
    def enterCcField(self, ctx:LLVMIRParser.CcFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#ccField.
    def exitCcField(self, ctx:LLVMIRParser.CcFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#alignField.
    def enterAlignField(self, ctx:LLVMIRParser.AlignFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#alignField.
    def exitAlignField(self, ctx:LLVMIRParser.AlignFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#allocatedField.
    def enterAllocatedField(self, ctx:LLVMIRParser.AllocatedFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#allocatedField.
    def exitAllocatedField(self, ctx:LLVMIRParser.AllocatedFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#annotationsField.
    def enterAnnotationsField(self, ctx:LLVMIRParser.AnnotationsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#annotationsField.
    def exitAnnotationsField(self, ctx:LLVMIRParser.AnnotationsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#argField.
    def enterArgField(self, ctx:LLVMIRParser.ArgFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#argField.
    def exitArgField(self, ctx:LLVMIRParser.ArgFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#associatedField.
    def enterAssociatedField(self, ctx:LLVMIRParser.AssociatedFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#associatedField.
    def exitAssociatedField(self, ctx:LLVMIRParser.AssociatedFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#attributesField.
    def enterAttributesField(self, ctx:LLVMIRParser.AttributesFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#attributesField.
    def exitAttributesField(self, ctx:LLVMIRParser.AttributesFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#baseTypeField.
    def enterBaseTypeField(self, ctx:LLVMIRParser.BaseTypeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#baseTypeField.
    def exitBaseTypeField(self, ctx:LLVMIRParser.BaseTypeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#checksumField.
    def enterChecksumField(self, ctx:LLVMIRParser.ChecksumFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#checksumField.
    def exitChecksumField(self, ctx:LLVMIRParser.ChecksumFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#checksumkindField.
    def enterChecksumkindField(self, ctx:LLVMIRParser.ChecksumkindFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#checksumkindField.
    def exitChecksumkindField(self, ctx:LLVMIRParser.ChecksumkindFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#columnField.
    def enterColumnField(self, ctx:LLVMIRParser.ColumnFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#columnField.
    def exitColumnField(self, ctx:LLVMIRParser.ColumnFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#configMacrosField.
    def enterConfigMacrosField(self, ctx:LLVMIRParser.ConfigMacrosFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#configMacrosField.
    def exitConfigMacrosField(self, ctx:LLVMIRParser.ConfigMacrosFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#containingTypeField.
    def enterContainingTypeField(self, ctx:LLVMIRParser.ContainingTypeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#containingTypeField.
    def exitContainingTypeField(self, ctx:LLVMIRParser.ContainingTypeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#countField.
    def enterCountField(self, ctx:LLVMIRParser.CountFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#countField.
    def exitCountField(self, ctx:LLVMIRParser.CountFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#debugInfoForProfilingField.
    def enterDebugInfoForProfilingField(self, ctx:LLVMIRParser.DebugInfoForProfilingFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#debugInfoForProfilingField.
    def exitDebugInfoForProfilingField(self, ctx:LLVMIRParser.DebugInfoForProfilingFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#declarationField.
    def enterDeclarationField(self, ctx:LLVMIRParser.DeclarationFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#declarationField.
    def exitDeclarationField(self, ctx:LLVMIRParser.DeclarationFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#directoryField.
    def enterDirectoryField(self, ctx:LLVMIRParser.DirectoryFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#directoryField.
    def exitDirectoryField(self, ctx:LLVMIRParser.DirectoryFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#discriminatorField.
    def enterDiscriminatorField(self, ctx:LLVMIRParser.DiscriminatorFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#discriminatorField.
    def exitDiscriminatorField(self, ctx:LLVMIRParser.DiscriminatorFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#dataLocationField.
    def enterDataLocationField(self, ctx:LLVMIRParser.DataLocationFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#dataLocationField.
    def exitDataLocationField(self, ctx:LLVMIRParser.DataLocationFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#discriminatorIntField.
    def enterDiscriminatorIntField(self, ctx:LLVMIRParser.DiscriminatorIntFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#discriminatorIntField.
    def exitDiscriminatorIntField(self, ctx:LLVMIRParser.DiscriminatorIntFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#dwarfAddressSpaceField.
    def enterDwarfAddressSpaceField(self, ctx:LLVMIRParser.DwarfAddressSpaceFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#dwarfAddressSpaceField.
    def exitDwarfAddressSpaceField(self, ctx:LLVMIRParser.DwarfAddressSpaceFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#dwoIdField.
    def enterDwoIdField(self, ctx:LLVMIRParser.DwoIdFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#dwoIdField.
    def exitDwoIdField(self, ctx:LLVMIRParser.DwoIdFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#elementsField.
    def enterElementsField(self, ctx:LLVMIRParser.ElementsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#elementsField.
    def exitElementsField(self, ctx:LLVMIRParser.ElementsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#emissionKindField.
    def enterEmissionKindField(self, ctx:LLVMIRParser.EmissionKindFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#emissionKindField.
    def exitEmissionKindField(self, ctx:LLVMIRParser.EmissionKindFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#encodingField.
    def enterEncodingField(self, ctx:LLVMIRParser.EncodingFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#encodingField.
    def exitEncodingField(self, ctx:LLVMIRParser.EncodingFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#entityField.
    def enterEntityField(self, ctx:LLVMIRParser.EntityFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#entityField.
    def exitEntityField(self, ctx:LLVMIRParser.EntityFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#enumsField.
    def enterEnumsField(self, ctx:LLVMIRParser.EnumsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#enumsField.
    def exitEnumsField(self, ctx:LLVMIRParser.EnumsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#exportSymbolsField.
    def enterExportSymbolsField(self, ctx:LLVMIRParser.ExportSymbolsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#exportSymbolsField.
    def exitExportSymbolsField(self, ctx:LLVMIRParser.ExportSymbolsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#exprField.
    def enterExprField(self, ctx:LLVMIRParser.ExprFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#exprField.
    def exitExprField(self, ctx:LLVMIRParser.ExprFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#extraDataField.
    def enterExtraDataField(self, ctx:LLVMIRParser.ExtraDataFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#extraDataField.
    def exitExtraDataField(self, ctx:LLVMIRParser.ExtraDataFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#fileField.
    def enterFileField(self, ctx:LLVMIRParser.FileFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#fileField.
    def exitFileField(self, ctx:LLVMIRParser.FileFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#filenameField.
    def enterFilenameField(self, ctx:LLVMIRParser.FilenameFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#filenameField.
    def exitFilenameField(self, ctx:LLVMIRParser.FilenameFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#flagsStringField.
    def enterFlagsStringField(self, ctx:LLVMIRParser.FlagsStringFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#flagsStringField.
    def exitFlagsStringField(self, ctx:LLVMIRParser.FlagsStringFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#getterField.
    def enterGetterField(self, ctx:LLVMIRParser.GetterFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#getterField.
    def exitGetterField(self, ctx:LLVMIRParser.GetterFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#globalsField.
    def enterGlobalsField(self, ctx:LLVMIRParser.GlobalsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#globalsField.
    def exitGlobalsField(self, ctx:LLVMIRParser.GlobalsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#identifierField.
    def enterIdentifierField(self, ctx:LLVMIRParser.IdentifierFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#identifierField.
    def exitIdentifierField(self, ctx:LLVMIRParser.IdentifierFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#importsField.
    def enterImportsField(self, ctx:LLVMIRParser.ImportsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#importsField.
    def exitImportsField(self, ctx:LLVMIRParser.ImportsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#includePathField.
    def enterIncludePathField(self, ctx:LLVMIRParser.IncludePathFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#includePathField.
    def exitIncludePathField(self, ctx:LLVMIRParser.IncludePathFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#inlinedAtField.
    def enterInlinedAtField(self, ctx:LLVMIRParser.InlinedAtFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#inlinedAtField.
    def exitInlinedAtField(self, ctx:LLVMIRParser.InlinedAtFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#isDeclField.
    def enterIsDeclField(self, ctx:LLVMIRParser.IsDeclFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#isDeclField.
    def exitIsDeclField(self, ctx:LLVMIRParser.IsDeclFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#isDefinitionField.
    def enterIsDefinitionField(self, ctx:LLVMIRParser.IsDefinitionFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#isDefinitionField.
    def exitIsDefinitionField(self, ctx:LLVMIRParser.IsDefinitionFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#isImplicitCodeField.
    def enterIsImplicitCodeField(self, ctx:LLVMIRParser.IsImplicitCodeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#isImplicitCodeField.
    def exitIsImplicitCodeField(self, ctx:LLVMIRParser.IsImplicitCodeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#isLocalField.
    def enterIsLocalField(self, ctx:LLVMIRParser.IsLocalFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#isLocalField.
    def exitIsLocalField(self, ctx:LLVMIRParser.IsLocalFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#isOptimizedField.
    def enterIsOptimizedField(self, ctx:LLVMIRParser.IsOptimizedFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#isOptimizedField.
    def exitIsOptimizedField(self, ctx:LLVMIRParser.IsOptimizedFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#isUnsignedField.
    def enterIsUnsignedField(self, ctx:LLVMIRParser.IsUnsignedFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#isUnsignedField.
    def exitIsUnsignedField(self, ctx:LLVMIRParser.IsUnsignedFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#apiNotesField.
    def enterApiNotesField(self, ctx:LLVMIRParser.ApiNotesFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#apiNotesField.
    def exitApiNotesField(self, ctx:LLVMIRParser.ApiNotesFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#languageField.
    def enterLanguageField(self, ctx:LLVMIRParser.LanguageFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#languageField.
    def exitLanguageField(self, ctx:LLVMIRParser.LanguageFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#lineField.
    def enterLineField(self, ctx:LLVMIRParser.LineFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#lineField.
    def exitLineField(self, ctx:LLVMIRParser.LineFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#linkageNameField.
    def enterLinkageNameField(self, ctx:LLVMIRParser.LinkageNameFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#linkageNameField.
    def exitLinkageNameField(self, ctx:LLVMIRParser.LinkageNameFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#lowerBoundField.
    def enterLowerBoundField(self, ctx:LLVMIRParser.LowerBoundFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#lowerBoundField.
    def exitLowerBoundField(self, ctx:LLVMIRParser.LowerBoundFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#macrosField.
    def enterMacrosField(self, ctx:LLVMIRParser.MacrosFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#macrosField.
    def exitMacrosField(self, ctx:LLVMIRParser.MacrosFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#nameTableKindField.
    def enterNameTableKindField(self, ctx:LLVMIRParser.NameTableKindFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#nameTableKindField.
    def exitNameTableKindField(self, ctx:LLVMIRParser.NameTableKindFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#nodesField.
    def enterNodesField(self, ctx:LLVMIRParser.NodesFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#nodesField.
    def exitNodesField(self, ctx:LLVMIRParser.NodesFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#offsetField.
    def enterOffsetField(self, ctx:LLVMIRParser.OffsetFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#offsetField.
    def exitOffsetField(self, ctx:LLVMIRParser.OffsetFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#producerField.
    def enterProducerField(self, ctx:LLVMIRParser.ProducerFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#producerField.
    def exitProducerField(self, ctx:LLVMIRParser.ProducerFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#rangesBaseAddressField.
    def enterRangesBaseAddressField(self, ctx:LLVMIRParser.RangesBaseAddressFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#rangesBaseAddressField.
    def exitRangesBaseAddressField(self, ctx:LLVMIRParser.RangesBaseAddressFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#rankField.
    def enterRankField(self, ctx:LLVMIRParser.RankFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#rankField.
    def exitRankField(self, ctx:LLVMIRParser.RankFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#retainedNodesField.
    def enterRetainedNodesField(self, ctx:LLVMIRParser.RetainedNodesFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#retainedNodesField.
    def exitRetainedNodesField(self, ctx:LLVMIRParser.RetainedNodesFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#retainedTypesField.
    def enterRetainedTypesField(self, ctx:LLVMIRParser.RetainedTypesFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#retainedTypesField.
    def exitRetainedTypesField(self, ctx:LLVMIRParser.RetainedTypesFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#runtimeLangField.
    def enterRuntimeLangField(self, ctx:LLVMIRParser.RuntimeLangFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#runtimeLangField.
    def exitRuntimeLangField(self, ctx:LLVMIRParser.RuntimeLangFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#runtimeVersionField.
    def enterRuntimeVersionField(self, ctx:LLVMIRParser.RuntimeVersionFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#runtimeVersionField.
    def exitRuntimeVersionField(self, ctx:LLVMIRParser.RuntimeVersionFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#scopeField.
    def enterScopeField(self, ctx:LLVMIRParser.ScopeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#scopeField.
    def exitScopeField(self, ctx:LLVMIRParser.ScopeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#scopeLineField.
    def enterScopeLineField(self, ctx:LLVMIRParser.ScopeLineFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#scopeLineField.
    def exitScopeLineField(self, ctx:LLVMIRParser.ScopeLineFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sdkField.
    def enterSdkField(self, ctx:LLVMIRParser.SdkFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sdkField.
    def exitSdkField(self, ctx:LLVMIRParser.SdkFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#setterField.
    def enterSetterField(self, ctx:LLVMIRParser.SetterFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#setterField.
    def exitSetterField(self, ctx:LLVMIRParser.SetterFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sizeField.
    def enterSizeField(self, ctx:LLVMIRParser.SizeFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sizeField.
    def exitSizeField(self, ctx:LLVMIRParser.SizeFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sourceField.
    def enterSourceField(self, ctx:LLVMIRParser.SourceFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sourceField.
    def exitSourceField(self, ctx:LLVMIRParser.SourceFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#spFlagsField.
    def enterSpFlagsField(self, ctx:LLVMIRParser.SpFlagsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#spFlagsField.
    def exitSpFlagsField(self, ctx:LLVMIRParser.SpFlagsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#splitDebugFilenameField.
    def enterSplitDebugFilenameField(self, ctx:LLVMIRParser.SplitDebugFilenameFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#splitDebugFilenameField.
    def exitSplitDebugFilenameField(self, ctx:LLVMIRParser.SplitDebugFilenameFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#splitDebugInliningField.
    def enterSplitDebugInliningField(self, ctx:LLVMIRParser.SplitDebugInliningFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#splitDebugInliningField.
    def exitSplitDebugInliningField(self, ctx:LLVMIRParser.SplitDebugInliningFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#strideField.
    def enterStrideField(self, ctx:LLVMIRParser.StrideFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#strideField.
    def exitStrideField(self, ctx:LLVMIRParser.StrideFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#stringLengthField.
    def enterStringLengthField(self, ctx:LLVMIRParser.StringLengthFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#stringLengthField.
    def exitStringLengthField(self, ctx:LLVMIRParser.StringLengthFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#stringLengthExpressionField.
    def enterStringLengthExpressionField(self, ctx:LLVMIRParser.StringLengthExpressionFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#stringLengthExpressionField.
    def exitStringLengthExpressionField(self, ctx:LLVMIRParser.StringLengthExpressionFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#stringLocationExpressionField.
    def enterStringLocationExpressionField(self, ctx:LLVMIRParser.StringLocationExpressionFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#stringLocationExpressionField.
    def exitStringLocationExpressionField(self, ctx:LLVMIRParser.StringLocationExpressionFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#sysrootField.
    def enterSysrootField(self, ctx:LLVMIRParser.SysrootFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#sysrootField.
    def exitSysrootField(self, ctx:LLVMIRParser.SysrootFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#targetFuncNameField.
    def enterTargetFuncNameField(self, ctx:LLVMIRParser.TargetFuncNameFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#targetFuncNameField.
    def exitTargetFuncNameField(self, ctx:LLVMIRParser.TargetFuncNameFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#templateParamsField.
    def enterTemplateParamsField(self, ctx:LLVMIRParser.TemplateParamsFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#templateParamsField.
    def exitTemplateParamsField(self, ctx:LLVMIRParser.TemplateParamsFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#thisAdjustmentField.
    def enterThisAdjustmentField(self, ctx:LLVMIRParser.ThisAdjustmentFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#thisAdjustmentField.
    def exitThisAdjustmentField(self, ctx:LLVMIRParser.ThisAdjustmentFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#thrownTypesField.
    def enterThrownTypesField(self, ctx:LLVMIRParser.ThrownTypesFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#thrownTypesField.
    def exitThrownTypesField(self, ctx:LLVMIRParser.ThrownTypesFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#typeMacinfoField.
    def enterTypeMacinfoField(self, ctx:LLVMIRParser.TypeMacinfoFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#typeMacinfoField.
    def exitTypeMacinfoField(self, ctx:LLVMIRParser.TypeMacinfoFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#typesField.
    def enterTypesField(self, ctx:LLVMIRParser.TypesFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#typesField.
    def exitTypesField(self, ctx:LLVMIRParser.TypesFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#unitField.
    def enterUnitField(self, ctx:LLVMIRParser.UnitFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#unitField.
    def exitUnitField(self, ctx:LLVMIRParser.UnitFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#upperBoundField.
    def enterUpperBoundField(self, ctx:LLVMIRParser.UpperBoundFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#upperBoundField.
    def exitUpperBoundField(self, ctx:LLVMIRParser.UpperBoundFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#valueIntField.
    def enterValueIntField(self, ctx:LLVMIRParser.ValueIntFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#valueIntField.
    def exitValueIntField(self, ctx:LLVMIRParser.ValueIntFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#valueStringField.
    def enterValueStringField(self, ctx:LLVMIRParser.ValueStringFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#valueStringField.
    def exitValueStringField(self, ctx:LLVMIRParser.ValueStringFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#varField.
    def enterVarField(self, ctx:LLVMIRParser.VarFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#varField.
    def exitVarField(self, ctx:LLVMIRParser.VarFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#virtualIndexField.
    def enterVirtualIndexField(self, ctx:LLVMIRParser.VirtualIndexFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#virtualIndexField.
    def exitVirtualIndexField(self, ctx:LLVMIRParser.VirtualIndexFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#virtualityField.
    def enterVirtualityField(self, ctx:LLVMIRParser.VirtualityFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#virtualityField.
    def exitVirtualityField(self, ctx:LLVMIRParser.VirtualityFieldContext):
        pass


    # Enter a parse tree produced by LLVMIRParser#vtableHolderField.
    def enterVtableHolderField(self, ctx:LLVMIRParser.VtableHolderFieldContext):
        pass

    # Exit a parse tree produced by LLVMIRParser#vtableHolderField.
    def exitVtableHolderField(self, ctx:LLVMIRParser.VtableHolderFieldContext):
        pass



del LLVMIRParser