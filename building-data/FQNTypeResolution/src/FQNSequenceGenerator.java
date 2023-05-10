package fqntypeparser;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.HashMap;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.*;

public class FQNSequenceGenerator extends ASTVisitor {
    private static final String SEPARATOR = "#";
    private String className, superClassName;
    private int offset;
    private int numOfExpressions = 0, numOfResolvedExpressions = 0;
    private int methodStartLine;
    private boolean correctTrailingNewLines;
    private CompilationUnit cu;
    private ArrayList<HashMap<String, String>> nodeInfo = new ArrayList<HashMap<String, String>>();

    public FQNSequenceGenerator(String className, String superClassName, int offset, int methodStartLine, boolean correctTrailingNewLines, CompilationUnit cu) {
        super(false);
        this.className = className;
        this.superClassName = superClassName;
        this.offset = offset;
        this.methodStartLine = methodStartLine;
        this.correctTrailingNewLines = correctTrailingNewLines;
        this.cu = cu;
    }

    public ArrayList<HashMap<String, String>> getNodeInfo() {
        return nodeInfo;
    }

    public int getCorrectedOffset(ASTNode node) {
        int nodeStartPosition = node.getStartPosition();
        if (correctTrailingNewLines) {
            int nodeLineOffset = cu.getLineNumber(nodeStartPosition) - methodStartLine;
            return nodeLineOffset;
        } else { 
            return 0;
        }
    }

    private Type getType(VariableDeclarationFragment node) {
        ASTNode p = node.getParent();
        if (p instanceof VariableDeclarationExpression)
            return ((VariableDeclarationExpression) p).getType();
        if (p instanceof VariableDeclarationStatement)
            return ((VariableDeclarationStatement) p).getType();
        return null;
    }

    private String getSignature(IMethodBinding method) {
        StringBuilder sb = new StringBuilder();
        sb.append(method.getDeclaringClass().getTypeDeclaration().getQualifiedName());
//        sb.append("." + method.getName());
        return sb.toString();
    }

    static String getUnresolvedType(Type type) {
        if (type.isArrayType()) {
            ArrayType t = (ArrayType) type;
            return getUnresolvedType(t.getElementType()) + getDimensions(t.getDimensions());
        } else if (type.isIntersectionType()) {
            IntersectionType it = (IntersectionType) type;
            @SuppressWarnings("unchecked")
            ArrayList<Type> types = new ArrayList<>(it.types());
            String s = getUnresolvedType(types.get(0));
            for (int i = 1; i < types.size(); i++)
                s += " & " + getUnresolvedType(types.get(i));
            return s;
        } else if (type.isParameterizedType()) {
            ParameterizedType t = (ParameterizedType) type;
            return getUnresolvedType(t.getType());
        } else if (type.isUnionType()) {
            UnionType it = (UnionType) type;
            @SuppressWarnings("unchecked")
            ArrayList<Type> types = new ArrayList<>(it.types());
            String s = getUnresolvedType(types.get(0));
            for (int i = 1; i < types.size(); i++)
                s += " | " + getUnresolvedType(types.get(i));
            return s;
        } else if (type.isNameQualifiedType()) {
            NameQualifiedType qt = (NameQualifiedType) type;
            return qt.getQualifier().getFullyQualifiedName() + "." + qt.getName().getIdentifier();
        } else if (type.isPrimitiveType()) {
            return type.toString();
        } else if (type.isQualifiedType()) {
            QualifiedType qt = (QualifiedType) type;
            return getUnresolvedType(qt.getQualifier()) + "." + qt.getName().getIdentifier();
        } else if (type.isSimpleType()) {
            return type.toString();
        } else if (type.isWildcardType()) {
            WildcardType wt = (WildcardType) type;
            String s = "?";
            if (wt.getBound() != null) {
                if (wt.isUpperBound())
                    s += "extends ";
                else
                    s += "super ";
                s += getUnresolvedType(wt.getBound());
            }
            return s;
        }
        
        return null;
    }

    private static String getDimensions(int dimensions) {
        String s = "";
        for (int i = 0; i < dimensions; i++)
            s += "[]";
        return s;
    }

    static String getResolvedType(Type type) {
        ITypeBinding tb = type.resolveBinding();
        if (tb == null || tb.isRecovered())
            return getUnresolvedType(type);
        tb = tb.getTypeDeclaration();
        if (tb.isLocal() || tb.getQualifiedName().isEmpty())
            return getUnresolvedType(type);
        if (type.isArrayType()) {
            ArrayType t = (ArrayType) type;
            return getResolvedType(t.getElementType()) + getDimensions(t.getDimensions());
        } else if (type.isIntersectionType()) {
            IntersectionType it = (IntersectionType) type;
            @SuppressWarnings("unchecked")
            ArrayList<Type> types = new ArrayList<>(it.types());
            String s = getResolvedType(types.get(0));
            for (int i = 1; i < types.size(); i++)
                s += " & " + getResolvedType(types.get(i));
            return s;
        } else if (type.isParameterizedType()) {
            ParameterizedType t = (ParameterizedType) type;
            return getResolvedType(t.getType());
        } else if (type.isUnionType()) {
            UnionType it = (UnionType) type;
            @SuppressWarnings("unchecked")
            ArrayList<Type> types = new ArrayList<>(it.types());
            String s = getResolvedType(types.get(0));
            for (int i = 1; i < types.size(); i++)
                s += " | " + getResolvedType(types.get(i));
            return s;
        } else if (type.isNameQualifiedType()) {
            return tb.getQualifiedName();
        } else if (type.isPrimitiveType()) {
            return type.toString();
        } else if (type.isQualifiedType()) {
            return tb.getQualifiedName();
        } else if (type.isSimpleType()) {
            return tb.getQualifiedName();
        } else if (type.isWildcardType()) {
            WildcardType wt = (WildcardType) type;
            String s = "?";
            if (wt.getBound() != null) {
                if (wt.isUpperBound())
                    s += "extends ";
                else
                    s += "super ";
                s += getResolvedType(wt.getBound());
            }
            return s;
        }
        
        return null;
    }

    @Override
    public void preVisit(ASTNode node) {
        if (node instanceof Expression) {
            numOfExpressions++;
            Expression e = (Expression) node;
            if (e.resolveTypeBinding() != null && !e.resolveTypeBinding().isRecovered())
                numOfResolvedExpressions++;
        } else if (node instanceof Statement) {
            if (node instanceof ConstructorInvocation) {
                numOfExpressions++;
                if (((ConstructorInvocation) node).resolveConstructorBinding() != null && !((ConstructorInvocation) node).resolveConstructorBinding().isRecovered())
                    numOfResolvedExpressions++;
            } else if (node instanceof SuperConstructorInvocation) {
                numOfExpressions++;
                if (((SuperConstructorInvocation) node).resolveConstructorBinding() != null && !((SuperConstructorInvocation) node).resolveConstructorBinding().isRecovered())
                    numOfResolvedExpressions++;
            }
        } else if (node instanceof Type) {
            numOfExpressions++;
            Type t = (Type) node;
            if (t.resolveBinding() != null && !t.resolveBinding().isRecovered())
                numOfResolvedExpressions++;
        }
    }

    @Override
    public boolean visit(ArrayAccess node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(ArrayCreation node) {
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }
        String utype = getUnresolvedType(node.getType()), rtype = getResolvedType(node.getType());

        if (startPosition != -1) {
            data.put("unresolvedType", utype.substring(0, utype.length() - 2));
            data.put("resolvedType", rtype.substring(0, rtype.length() - 2));
            data.put("nodeType", "ARRAY_CREATION");
            nodeInfo.add(data);
        }
        if (node.getInitializer() != null)
            node.getInitializer().accept(this);
        else
            for (int i = 0; i < node.dimensions().size(); i++)
                ((Expression) (node.dimensions().get(i))).accept(this);
        return false;
    }


    @Override
    public boolean visit(ArrayInitializer node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(AssertStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(Assignment node) {
        node.getLeftHandSide().accept(this);
        node.getRightHandSide().accept(this);
        return false;
    }

    @Override
    public boolean visit(Block node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(BooleanLiteral node) {
        return false;
    }

    @Override
    public boolean visit(BreakStatement node) {
        return false;
    }

    @Override
    public boolean visit(CastExpression node) {
        String utype = getUnresolvedType(node.getType()), rtype = getResolvedType(node.getType());
        node.getExpression().accept(this);

        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            HashMap<String, String> data = new HashMap<String, String>();
            data.put("debugText", node.toString());
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeType", "CAST_EXPRESSION");
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
            nodeInfo.add(data);
        }
        return false;
    }

    @Override
    public boolean visit(CatchClause node) {
        HashMap<String, String> data = new HashMap<String, String>();
        String utype = getUnresolvedType(node.getException().getType());
        String rtype = getResolvedType(node.getException().getType());

        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeType", "CATCH_CLAUSE");
            nodeInfo.add(data);
        }
        return super.visit(node);
    }

    @Override
    public boolean visit(CharacterLiteral node) {
        return false;
    }

    @Override
    public boolean visit(ClassInstanceCreation node) {
        ITypeBinding tb = node.getType().resolveBinding();
        if (tb != null && tb.getTypeDeclaration().isLocal())
            return false;
        String utype = getUnresolvedType(node.getType());
        String rtype = null;
        IMethodBinding b = node.resolveConstructorBinding();
        if (b == null) {
            rtype = utype;
        } else {
            rtype = getSignature(b.getMethodDeclaration());
            if (node.getParent() instanceof ThrowStatement)
                rtype = rtype.replace("." + b.getMethodDeclaration().getName(), "");
        }
        for (Iterator it = node.arguments().iterator(); it.hasNext(); ) {
            Expression e = (Expression) it.next();
            e.accept(this);
        }
        if (node.getAnonymousClassDeclaration() != null)
            node.getAnonymousClassDeclaration().accept(this);
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        String nodeArguments = "(";
        for (int i = 0; i < node.arguments().size(); i++) {
            nodeArguments += node.arguments().get(i);
            if (i < node.arguments().size() - 1)
                nodeArguments += ", ";
        }
        nodeArguments += ")";

        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            HashMap<String, String> data = new HashMap<String, String>();
            data.put("debugText", node.toString());
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeArguments", nodeArguments);
            data.put("nodeType", "CLASS_INSTANCE_CREATION");
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
            nodeInfo.add(data);
        }
        return false;
    }

    @Override
    public boolean visit(ConditionalExpression node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(ConstructorInvocation node) {
        IMethodBinding b = node.resolveConstructorBinding();
        ITypeBinding tb = null;
        if (b != null && b.getDeclaringClass() != null)
            tb = b.getDeclaringClass().getTypeDeclaration();
        if (tb != null) {
            if (tb.isLocal() || tb.getQualifiedName().isEmpty())
                return false;
        }
        String name = "." + className;
        String utype = name;
        if (tb != null)
            name = getSignature(b.getMethodDeclaration());
        String rtype = name;
        String nodeArguments = "(";
        for (int i = 0; i < node.arguments().size(); i++) {
            nodeArguments += node.arguments().get(i);
            if (i < node.arguments().size() - 1)
                nodeArguments += ", ";
        }
        nodeArguments += ")";
        
        if (utype == rtype) {
            if (utype.startsWith(".")) {
                utype = utype.substring(1, utype.length());
                rtype = rtype.substring(1, rtype.length());
            }
        }
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            HashMap<String, String> data = new HashMap<String, String>();
            data.put("debugText", node.toString());
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("partial", "<blank>" + utype + nodeArguments + ";");
            data.put("full", rtype + nodeArguments + ";");
            data.put("nodeType", "CONSTRUCTOR_INVOCATION");
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
            nodeInfo.add(data);
        }
        for (int i = 0; i < node.arguments().size(); i++) {
            ASTNode nodeArgument = (ASTNode) node.arguments().get(i);
/*            if (!(nodeArgument instanceof MethodInvocation)) */
                nodeArgument.accept(this);
        }
        return false;
    }

    @Override
    public boolean visit(ContinueStatement node) {
        return false;
    }

    @Override
    public boolean visit(CreationReference node) {
        return false;
    }

    @Override
    public boolean visit(Dimension node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(DoStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(EmptyStatement node) {
        return false;
    }

    @Override
    public boolean visit(EnhancedForStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(EnumConstantDeclaration node) {
        return false;
    }

    @Override
    public boolean visit(EnumDeclaration node) {
        return false;
    }

    @Override
    public boolean visit(ExpressionMethodReference node) {
        return false;
    }

    @Override
    public boolean visit(ExpressionStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(FieldAccess node) {
        IVariableBinding b = node.resolveFieldBinding();
        ITypeBinding tb = null;
        if (b != null) {
            tb = b.getDeclaringClass();
            if (tb != null) {
                tb = tb.getTypeDeclaration();
                if (tb.isLocal() || tb.getQualifiedName().isEmpty())
                    return false;
            }
        }
        node.getExpression().accept(this);
        String name = "." + node.getName().getIdentifier();
        String utype = name;
        if (b != null) {
            if (tb != null)
                name = getQualifiedName(tb.getTypeDeclaration()) + name;
        }
        String rtype = name;
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            HashMap<String, String> data = new HashMap<String, String>();
            data.put("debugText", node.toString());
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("partial", "<blank>" + utype);
            data.put("full", rtype);
            data.put("nodeType", "FIELD_ACCESS");
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
            nodeInfo.add(data);
        }

        return false;
    }

    @Override
    public boolean visit(FieldDeclaration node) {
        return false;
    }

    @Override
    public boolean visit(ForStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(IfStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(ImportDeclaration node) {
        return false;
    }

    @Override
    public boolean visit(InfixExpression node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(Initializer node) {
        return false;
    }

    @Override
    public boolean visit(InstanceofExpression node) {
        node.getLeftOperand().accept(this);
        String rtype = getResolvedType(node.getRightOperand()), utype = getUnresolvedType(node.getRightOperand());
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            HashMap<String, String> data = new HashMap<String, String>();
            data.put("debugText", node.toString());
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeType", "INSTANCEOF_EXPRESSION");
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
            nodeInfo.add(data);
        }

        return false;
    }

    @Override
    public boolean visit(LabeledStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(LambdaExpression node) {
        return false;
    }

    @Override
    public boolean visit(MethodDeclaration node) {
        for (int i = 0; i < node.parameters().size(); i++)
            ((SingleVariableDeclaration) node.parameters().get(i)).accept(this);

        if (node.getBody() != null && !node.getBody().statements().isEmpty())
            node.getBody().accept(this);
        return false;
    }

    @Override
    public boolean visit(MethodInvocation node) {
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        HashMap<String, String> data = new HashMap<String, String>();
        String utype = null, rtype = null;
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
        }

        if (node.getExpression() != null && node.getExpression() instanceof TypeLiteral) {
            TypeLiteral lit = (TypeLiteral) node.getExpression();
            utype = getUnresolvedType(lit.getType());
            rtype = getResolvedType(lit.getType());
            if (startPosition != -1) {
                utype = utype + "." + node.getName().getIdentifier();
                rtype = rtype + "." + node.getName().getIdentifier();
                data.put("unresolvedType", utype);
                data.put("resolvedType", rtype);
            }
        } else {
            IMethodBinding b = node.resolveMethodBinding();
            ITypeBinding tb = null;
            if (b != null) {
                tb = b.getDeclaringClass();
                if (tb != null) {
                    tb = tb.getTypeDeclaration();
                    if (tb.isLocal() || tb.getQualifiedName().isEmpty()) {
                        if (startPosition != -1) {
                            data.put("nodeType", "METHOD_INVOCATION");
                            nodeInfo.add(data);
                        }
                        return false;
                    }
                }
            }
            if (node.getExpression() != null) {
                node.getExpression().accept(this);
            } else {
                if (tb != null) {
                    if (startPosition != -1) {
                        utype = getName(tb);
                        rtype = getQualifiedName(tb);
                    }
                } else {
                    if (startPosition != -1) {
                        utype = "this";
                        rtype = "this";
                    }
                }
            }
            String name = "."+ node.getName().getIdentifier();
            if (startPosition != -1) {
                utype = name;
                data.put("unresolvedType", utype);
            }
            if (tb != null)
                name = getSignature(b.getMethodDeclaration());
            rtype = name;
        }
        for (int i = 0; i < node.arguments().size(); i++) {
            ASTNode nodeArgument = (ASTNode) node.arguments().get(i);
//            if (!(nodeArgument instanceof MethodInvocation))
                nodeArgument.accept(this);
        }
        String nodeArguments = "(";
        for (int i = 0; i < node.arguments().size(); i++) {
            nodeArguments += node.arguments().get(i);
            if (i < node.arguments().size() - 1)
                nodeArguments += ", ";
        }
        nodeArguments += ")";
        if (startPosition != -1) {
            data.put("resolvedType", rtype);
            data.put("nodeArguments", nodeArguments);
            data.put("nodeType", "METHOD_INVOCATION");
            nodeInfo.add(data);
        }
        return false;
    }

    @Override
    public boolean visit(Modifier node) {
        return false;
    }

    @Override
    public boolean visit(NormalAnnotation node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(NullLiteral node) {
        return false;
    }

    @Override
    public boolean visit(NumberLiteral node) {
        return false;
    }

    @Override
    public boolean visit(PackageDeclaration node) {
        return false;
    }

    @Override
    public boolean visit(ParenthesizedExpression node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(PostfixExpression node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(PrefixExpression node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(QualifiedName node) {
        String utype = null, rtype = null;
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        HashMap<String, String> data = new HashMap<String, String>();
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }

        IBinding b = node.resolveBinding();
        IVariableBinding vb = null;
        ITypeBinding tb = null;
        if (b != null) {
            if (b instanceof IVariableBinding) {
                vb = (IVariableBinding) b;
                tb = vb.getDeclaringClass();
                if (tb != null) {
                    tb = tb.getTypeDeclaration();
                    if (tb.isLocal() || tb.getQualifiedName().isEmpty())
                        if (startPosition != -1) {
                            data.put("unresolvedType", "" + utype);
                            data.put("resolvedType", "" + rtype);
                            data.put("nodeType", "QUALIFIED_NAME");
                            nodeInfo.add(data);
                        }
                        return false;
                }
            } else if (b instanceof ITypeBinding) {
                tb = ((ITypeBinding) b).getTypeDeclaration();
                if (tb.isLocal() || tb.getQualifiedName().isEmpty()) {
                    if (startPosition != -1) {
                        data.put("unresolvedType", "" + utype);
                        data.put("resolvedType", "" + rtype);
                        data.put("nodeType", "QUALIFIED_NAME");
                        nodeInfo.add(data);
                    }
                    return false;
                }
                utype = node.getFullyQualifiedName();
                rtype = getQualifiedName(tb);
                if (startPosition != -1) {
                    data.put("unresolvedType", "" + utype);
                    data.put("resolvedType", "" + rtype);
                    data.put("nodeType", "QUALIFIED_NAME");
                    nodeInfo.add(data);
                }
                return false;
            }
        } else {
            utype = node.getFullyQualifiedName();
            rtype = node.getFullyQualifiedName();
            if (startPosition != -1) {
                data.put("unresolvedType", "" + utype);
                data.put("resolvedType", "" + rtype);
                data.put("nodeType", "QUALIFIED_NAME");
                nodeInfo.add(data);
            }
            return false;
        }
        node.getQualifier().accept(this);
        String name = "." + node.getName().getIdentifier();
        utype = name;
        if (b != null) {
            if (b instanceof IVariableBinding) {
                if (tb != null)
                    name = getQualifiedName(tb.getTypeDeclaration()) + name;
            }
        }
        rtype = name;
        if (startPosition != -1) {
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeType", "QUALIFIED_NAME");
            nodeInfo.add(data);
        }
        return false;
    }

    @Override
    public boolean visit(ReturnStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(SimpleName node) {
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        HashMap<String, String> data = new HashMap<String, String>();
        if (startPosition == -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }

        IBinding b = node.resolveBinding();
        if (b != null) {
            if (b instanceof IVariableBinding) {
                IVariableBinding vb = (IVariableBinding) b;
                ITypeBinding tb = vb.getType();
                if (tb != null) {
                    tb = tb.getTypeDeclaration();
                    if (tb.isLocal() || tb.getQualifiedName().isEmpty()) {
                        if (startPosition == -1)
                            nodeInfo.add(data);
                        return false;
                    }
                    if (startPosition == -1) {
                        data.put("unresolvedType", getQualifiedName(tb));
                        data.put("resolvedType", getName(tb));
                        data.put("nodeType", "SIMPLE_NAME");
                    }
                }
            } else if (b instanceof ITypeBinding) {
                ITypeBinding tb = (ITypeBinding) b;
                tb = tb.getTypeDeclaration();
                if (tb.isLocal() || tb.getQualifiedName().isEmpty()) {
                    if (startPosition != -1)
                        nodeInfo.add(data);
                    return false;
                }
                if (startPosition == -1) {
                    data.put("unresolvedType", getQualifiedName(tb));
                    data.put("resolvedType", getName(tb));
                    data.put("nodeType", "SIMPLE_NAME");
                }
            }
        } else {
            if (startPosition == -1) {
                data.put("unresolvedType", node.getIdentifier());
                data.put("resolvedType", node.getIdentifier());
                data.put("nodeType", "SIMPLE_NAME");
            }
        }
        if (startPosition == -1)
            nodeInfo.add(data);
        return false;
    }

    @Override
    public boolean visit(SingleMemberAnnotation node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(SingleVariableDeclaration node) {
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }

        ITypeBinding tb = node.getType().resolveBinding();
        if (tb != null && tb.getTypeDeclaration().isLocal()) {
            if (startPosition != -1)
                nodeInfo.add(data);
            return false;
        }
        String utype = getUnresolvedType(node.getType()), rtype = getResolvedType(node.getType());
        if (startPosition != -1) {
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeType", "SINGLE_VARIABLE_DECLARATION");
        }
        if (node.getInitializer() != null) {
            node.getInitializer().accept(this);
        }
        if (startPosition != -1)
            nodeInfo.add(data);
        return false;
    }

    @Override
    public boolean visit(StringLiteral node) {
        return false;
    }

    @Override
    public boolean visit(SuperConstructorInvocation node) {
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        String utype = null, rtype = null;
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }

        IMethodBinding b = node.resolveConstructorBinding();
        ITypeBinding tb = null;
        if (b != null && b.getDeclaringClass() != null)
            tb = b.getDeclaringClass().getTypeDeclaration();
        if (tb != null) {
            if (tb.isLocal() || tb.getQualifiedName().isEmpty()) {
                if (startPosition != -1)
                    nodeInfo.add(data);
                return false;
            }
        }
        String name = "." + superClassName;
        if (startPosition != -1) {
            utype = name;
            data.put("unresolvedType", utype);
        }
        if (tb != null)
            name = getSignature(b.getMethodDeclaration());
        rtype = name;
        String nodeArguments = "(";
        for (int i = 0; i < node.arguments().size(); i++) {
            nodeArguments += node.arguments().get(i);
            if (i < node.arguments().size() - 1)
                nodeArguments += ", ";
        }
        nodeArguments += ")";
        if (startPosition != -1) {
            data.put("resolvedType", rtype);
            data.put("partial", "<blank>" + nodeArguments + ";");
            data.put("full", rtype + nodeArguments + ";");
            data.put("nodeType", "SUPER_CONSTRUCTOR_INVOCATION");
            nodeInfo.add(data);
        }
        for (int i = 0; i < node.arguments().size(); i++) {
            ASTNode nodeArgument = (ASTNode) node.arguments().get(i);
/*            if (!(nodeArgument instanceof MethodInvocation)) */
                nodeArgument.accept(this);
        }
        return false;
    }

    @Override
    public boolean visit(SuperFieldAccess node) {
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }

        IVariableBinding b = node.resolveFieldBinding();
        ITypeBinding tb = null;
        if (b != null && b.getDeclaringClass() != null) {
            tb = b.getDeclaringClass().getTypeDeclaration();
            if (tb.isLocal() || tb.getQualifiedName().isEmpty()) {
                if (startPosition != -1)
                    nodeInfo.add(data);
                return false;
            }
            String utype = getName(tb), rtype = getQualifiedName(tb);
            if (startPosition != -1) {
                data.put("unresolvedType", "" + utype);
                data.put("resolvedType", "" + rtype);
                data.put("nodeType", "SUPER_FIELD_ACCESS");
            }
        } else {
            String utype = "super", rtype = "super";
            if (startPosition != -1) {
                data.put("unresolvedType", utype);
                data.put("resolvedType", rtype);
                data.put("nodeType", "SUPER_FIELD_ACCESS");
            }
        }
        String name = "." + node.getName().getIdentifier();
        String utype = name;
        if (startPosition != -1)
            data.put("unresolvedType", utype);
        if (tb != null)
            name = getQualifiedName(tb) + name;
        String rtype = name;
        if (startPosition != -1) {
            data.put("resolvedType", rtype);
            data.put("partial", "<blank>" + utype);
            data.put("full", rtype);
            data.put("nodeType", "SUPER_FIELD_ACCESS");
            nodeInfo.add(data);
        }
        return false;
    }

    @Override
    public boolean visit(SuperMethodInvocation node) {
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        String utype = null, rtype = null;
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }

        IMethodBinding b = node.resolveMethodBinding();
        ITypeBinding tb = null;
        if (b != null && b.getDeclaringClass() != null)
            tb = b.getDeclaringClass().getTypeDeclaration();
        if (tb != null) {
            if (tb.isLocal() || tb.getQualifiedName().isEmpty()) {
                if (startPosition != -1)
                    nodeInfo.add(data);
                return false;
            }
            if (startPosition != -1) {
                utype = getName(tb);
                rtype = getQualifiedName(tb);
                data.put("unresolvedType", utype);
                data.put("resolvedType", rtype);
                data.put("nodeType", "SUPER_METHOD_INVOCATION");
            }
        } else {
            if (startPosition != -1) {
                utype = "super";
                rtype = "super";
                data.put("unresolvedType", utype);
                data.put("resolvedType", rtype);
                data.put("nodeType", "SUPER_METHOD_INVOCATION");
            }
        }
        String name = "." + node.getName().getIdentifier();
        if (startPosition != -1) {
            utype = name;
            data.put("unresolvedType", utype);

        }
        String nodeArguments = "(";
        for (int i = 0; i < node.arguments().size(); i++) {
            nodeArguments += node.arguments().get(i);
            if (i < node.arguments().size() - 1)
                nodeArguments += ", ";
        }
        nodeArguments += ")";

        if (tb != null)
            name = getSignature(b.getMethodDeclaration());
        rtype = name;
        if (startPosition != -1) {
            data.put("resolvedType", rtype);
            data.put("nodeArguments", nodeArguments);
            nodeInfo.add(data);
        }
        for (int i = 0; i < node.arguments().size(); i++) {
            ASTNode nodeArgument = (ASTNode) node.arguments().get(i);
/*            if (!(nodeArgument instanceof MethodInvocation)) */
                nodeArgument.accept(this);
        }
        return false;
    }

    @Override
    public boolean visit(SuperMethodReference node) {
        return false;
    }

    @Override
    public boolean visit(SwitchCase node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(SwitchStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(SynchronizedStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(ThisExpression node) {
        ITypeBinding b = node.resolveTypeBinding();
        if (b != null) {
            b = b.getTypeDeclaration();
            if (b.isLocal() || b.getQualifiedName().isEmpty())
                return false;
        }
        return false;
    }

    @Override
    public boolean visit(ThrowStatement node) {
        String utype = null, rtype = null;
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (node.getExpression().getClass().getSimpleName().toString().equals("SimpleName")) {
            SimpleName simpleNameNode = (SimpleName)node.getExpression();
            IBinding b = simpleNameNode.resolveBinding();
            if (b != null) {
                if (b instanceof IVariableBinding) {
                    IVariableBinding vb = (IVariableBinding) b;
                    ITypeBinding tb = vb.getType();
                    if (tb != null) {
                        tb = tb.getTypeDeclaration();
                        utype = getName(tb);
                        rtype = getQualifiedName(tb);
                    }
                } else if (b instanceof ITypeBinding) {
                    ITypeBinding tb = (ITypeBinding) b;
                    tb = tb.getTypeDeclaration();
                    utype = getName(tb);
                    rtype = getQualifiedName(tb);
                }
            } else {
                utype = simpleNameNode.getIdentifier();
                rtype = simpleNameNode.getIdentifier();
            }
            if (utype.equals(rtype)) {
                if (startPosition != -1) {
                    data.put("unresolvedType", utype);
                    data.put("resolvedType", rtype);
                }
            } else {
                if (startPosition != -1) {
                    data.put("unresolvedType", utype);
                    data.put("resolvedType", rtype);
                    data.put("partial", "throw <blank>");
                    data.put("full", "throw " + rtype);
                }
            }
        } else {
            if (startPosition != -1) {
                data.put("unresolvedType", null);
                data.put("resolvedType", null);
            }
        }
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("nodeType", "THROW_STATEMENT");
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
            nodeInfo.add(data);
        }
        return super.visit(node);
    }

    @Override
    public boolean visit(TryStatement node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(TypeDeclaration node) {
        return false;
    }

    @Override
    public boolean visit(TypeDeclarationStatement node) {
        return false;
    }

    @Override
    public boolean visit(TypeLiteral node) {
        return false;
    }

    @Override
    public boolean visit(TypeMethodReference node) {
        return super.visit(node);
    }

    @Override
    public boolean visit(TypeParameter node) {
        return super.visit(node);
    }
    
    @Override
    public boolean visit(VariableDeclarationExpression node) {
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }

        ITypeBinding tb = node.getType().resolveBinding();
        if (tb != null && tb.getTypeDeclaration().isLocal()) {
            if (startPosition != -1) {
                nodeInfo.add(data);
            }
            return false;
        }
        String utype = getUnresolvedType(node.getType()), rtype = getResolvedType(node.getType());
        if (startPosition != -1) {
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeType", "VARIABLE_DECLARATION_EXPRESSION");
            nodeInfo.add(data);
        }
        for (int i = 0; i < node.fragments().size(); i++)
            ((ASTNode) node.fragments().get(i)).accept(this);
        return false;
    }

    @Override
    public boolean visit(VariableDeclarationStatement node) {
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }

        ITypeBinding tb = node.getType().resolveBinding();
        if (tb != null && tb.getTypeDeclaration().isLocal()) {
            if (startPosition != -1) {
                nodeInfo.add(data);
            }
            return false;
        }
        String utype = getUnresolvedType(node.getType()), rtype = getResolvedType(node.getType());
        if (startPosition != -1) {
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeType", "VARIABLE_DECLARATION_STATEMENT");
            nodeInfo.add(data);
        }
        for (int i = 0; i < node.fragments().size(); i++)
            ((ASTNode) node.fragments().get(i)).accept(this);
        return false;
    }

    @Override
    public boolean visit(VariableDeclarationFragment node) {
        HashMap<String, String> data = new HashMap<String, String>();
        int startPosition = node.getStartPosition();
        int correctedOffset = getCorrectedOffset(node);
        if (startPosition != -1) {
            startPosition = startPosition - correctedOffset;
            int endPosition = startPosition + node.getLength();
            if (correctTrailingNewLines)
                endPosition += 1;
            data.put("debugText", node.toString());
            data.put("start", "" + (startPosition - offset - 1));
            data.put("end", "" + (endPosition - offset - 1));
        }
        Type type = getType(node);
        String utype = getUnresolvedType(type), rtype = getResolvedType(type);
        if (startPosition != -1) {
            data.put("unresolvedType", utype);
            data.put("resolvedType", rtype);
            data.put("nodeType", "VARIABLE_DECLARATION_FRAGMENT");
            nodeInfo.add(data);
        }
        if (node.getInitializer() != null) {
            node.getInitializer().accept(this);
        }
        return false;
    }

    @Override
    public boolean visit(WhileStatement node) {
        return super.visit(node);
    }
    
    @Override
    public boolean visit(ArrayType node) {
        return false;
    }
    
    @Override
    public boolean visit(IntersectionType node) {
        return false;
    }
    
    @Override
    public boolean visit(ParameterizedType node) {
        return false;
    }
    
    @Override
    public boolean visit(UnionType node) {
        return false;
    }
    
    @Override
    public boolean visit(NameQualifiedType node) {
        return false;
    }
    
    @Override
    public boolean visit(PrimitiveType node) {
        return false;
    }
    
    @Override
    public boolean visit(QualifiedType node) {
        return false;
    }
    
    @Override
    public boolean visit(SimpleType node) {
        return false;
    }
    
    @Override
    public boolean visit(WildcardType node) {
        return false;
    }

    private String getQualifiedName(ITypeBinding tb) {
        if (tb.isArray())
            return getQualifiedName(tb.getComponentType().getTypeDeclaration()) + getDimensions(tb.getDimensions());
        return tb.getQualifiedName();
    }

    private String getName(ITypeBinding tb) {
        if (tb.isArray())
            return getName(tb.getComponentType().getTypeDeclaration()) + getDimensions(tb.getDimensions());
        return tb.getName();
    }

}
