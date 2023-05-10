package fqntypeparser;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.*;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.FileASTRequestor;
import org.eclipse.jdt.core.dom.ImportDeclaration;
import org.eclipse.jdt.core.dom.MethodDeclaration;
import org.eclipse.jdt.core.dom.SingleVariableDeclaration;
import org.eclipse.jdt.core.dom.TypeDeclaration;

import org.json.simple.JSONObject;

import org.apache.commons.io.FileUtils;

import fqntypeparser.ClassPathUtil.PomFile;
import fqntypeparser.FileUtil;

public class ProjectSequencesGenerator {
    private static final boolean PARSE_INDIVIDUAL_SRC = false, SCAN_FILES_FRIST = false;
    
    private String inPath, outPath, libFolderPath;
    private boolean testing = false;
    private HashSet<String> badFiles = new HashSet<>();
    
    public ProjectSequencesGenerator(String inPath, String libFolderPath) {
        this.inPath = inPath;
		this.libFolderPath = libFolderPath;
    }
    
    public ProjectSequencesGenerator(String inPath, String libFolderPath, boolean testing) {
        this(inPath, libFolderPath);
        this.testing = testing;
    }

    public int  generateSequences(String outPath) {
        return generateSequences(true, null, outPath);
    }

    public int generateSequences(final boolean keepUnresolvables, final String lib, final String outPath) {
        this.outPath = outPath;
        String[] jarPaths = getJarPaths(this.libFolderPath);
        ArrayList<String> rootPaths = getRootPaths();
        new File(outPath).mkdirs();
        int numOfSequences = 0;

        for (String rootPath : rootPaths) {
            String[] sourcePaths = getSourcePaths(rootPath, new String[]{".java"});
            
            @SuppressWarnings("rawtypes")
            Map options = JavaCore.getOptions();
            options.put(JavaCore.COMPILER_COMPLIANCE, JavaCore.VERSION_1_8);
            options.put(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, JavaCore.VERSION_1_8);
            options.put(JavaCore.COMPILER_SOURCE, JavaCore.VERSION_1_8);
            ASTParser parser = ASTParser.newParser(AST.JLS8);
            parser.setCompilerOptions(options);
            parser.setEnvironment(jarPaths, new String[]{}, new String[]{}, true);
            parser.setResolveBindings(true);
            parser.setBindingsRecovery(false);
            
            StatTypeFileASTRequestor r = new StatTypeFileASTRequestor(keepUnresolvables, lib);
            try {
                parser.createASTs(sourcePaths, null, new String[0], r, null);
            } catch (Throwable t) {
                if (testing) {
                    System.err.println(t.getMessage());
                    t.printStackTrace();
                }
            }
            numOfSequences += r.numOfSequences;
        }
        return numOfSequences;
    }
    
    private class StatTypeFileASTRequestor extends FileASTRequestor {
        int numOfSequences = 0;
        private boolean keepUnresolvables;
        private String lib;
        
        public StatTypeFileASTRequestor(boolean keepUnresolvables, String lib) {
            this.keepUnresolvables = keepUnresolvables;
            this.lib = lib;
        }

        @Override
        public void acceptAST(String sourceFilePath, CompilationUnit ast) {
            if (ast.getPackage() == null)
                return;

            if (lib != null) {
                if (lib == "java") {
                    boolean hasLib = false;
                    String packageName = ast.getPackage().getName().getFullyQualifiedName();
                    if (packageName.startsWith("java") || packageName.startsWith(" jdk") || packageName.startsWith("sun"))
                        hasLib = true;
                    if (!hasLib && ast.imports() != null) {
                        for (int i = 0; i < ast.imports().size(); i++) {
                            ImportDeclaration ic = (ImportDeclaration) ast.imports().get(i);
                            String importFQN = ic.getName().getFullyQualifiedName();
                            if (importFQN.startsWith("java") || importFQN.startsWith("jdk") || importFQN.startsWith("sun")) {
                                hasLib = true;
                                break;
                            }
                        }
                    }
                    if (!hasLib)
                        return;
                } else {
                    boolean hasLib = false;
                    if (ast.getPackage().getName().getFullyQualifiedName().startsWith(lib))
                        hasLib = true;
                    if (!hasLib && ast.imports() != null) {
                        for (int i = 0; i < ast.imports().size(); i++) {
                            ImportDeclaration ic = (ImportDeclaration) ast.imports().get(i);
                            if (ic.getName().getFullyQualifiedName().startsWith(lib)) {
                                hasLib = true;
                                break;
                            }
                        }
                    }
                    if (!hasLib)
                        return;
                }
            }
            if (testing)
                System.out.println(sourceFilePath);
            for (int i = 0; i < ast.types().size(); i++) {
                if (ast.types().get(i) instanceof TypeDeclaration) {
                    TypeDeclaration td = (TypeDeclaration) ast.types().get(i);
                    numOfSequences += generateSequence(ast, keepUnresolvables, lib, td, sourceFilePath, "");
                }
            }
        }
    }

    private ArrayList<String> getRootPaths() {
        ArrayList<String> rootPaths = new ArrayList<>();
        if (PARSE_INDIVIDUAL_SRC)
            getRootPaths(new File(inPath), rootPaths);
        else {
            if (SCAN_FILES_FRIST)
                getRootPaths(new File(inPath), rootPaths);
            rootPaths = new ArrayList<>();
            rootPaths.add(inPath);
        }
        return rootPaths;
    }

    private void getRootPaths(File file, ArrayList<String> rootPaths) {
        if (file.isDirectory()) {
            for (File sub : file.listFiles())
                getRootPaths(sub, rootPaths);
        } else if (file.getName().endsWith(".java")) {
            Map options = JavaCore.getOptions();
            options.put(JavaCore.COMPILER_COMPLIANCE, JavaCore.VERSION_1_8);
            options.put(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, JavaCore.VERSION_1_8);
            options.put(JavaCore.COMPILER_SOURCE, JavaCore.VERSION_1_8);
            ASTParser parser = ASTParser.newParser(AST.JLS8);
            parser.setCompilerOptions(options);
            parser.setSource(FileUtil.getFileContent(file.getAbsolutePath()).toCharArray());
            try {
                CompilationUnit ast = (CompilationUnit) parser.createAST(null);
                if (ast.getPackage() != null && !ast.types().isEmpty() && ast.types().get(0) instanceof TypeDeclaration) {
                    String name = ast.getPackage().getName().getFullyQualifiedName();
//                    name = name.replace('.', '\\');
                    name = name.replace('.', '/');
                    String p = file.getParentFile().getAbsolutePath();
                    if (p.endsWith(name))
                        add(p.substring(0, p.length() - name.length() - 1), rootPaths);
                } /*else 
                    badFiles.add(file.getAbsolutePath());*/
            } catch (Throwable t) {
                badFiles.add(file.getAbsolutePath());
            }
        }
    }

    private void add(String path, ArrayList<String> rootPaths) {
        int index = Collections.binarySearch(rootPaths, path);
        if (index < 0) {
            index = - index - 1;
            int i = rootPaths.size() - 1;
            while (i > index) {
                if (rootPaths.get(i).startsWith(path))
                    rootPaths.remove(i);
                i--;
            }
            i = index - 1;
            while (i >= 0) {
                if (path.startsWith(rootPaths.get(i)))
                    return;
                i--;
            }
            rootPaths.add(index, path);
        }
    }

    private int generateSequence(CompilationUnit ast, boolean keepUnresolvables, String lib, TypeDeclaration td, String path, String outer) {
        int numOfSequences = 0;
        String name = outer.isEmpty() ? td.getName().getIdentifier() : outer + "." + td.getName().getIdentifier();
        String className = td.getName().getIdentifier(), superClassName = null;
        String packageName = ast.getPackage().getName().getFullyQualifiedName();
        String sourceCode = "";
        try {
            sourceCode = new String(Files.readAllBytes(Paths.get(path)));
        } catch (IOException ex) {
            System.out.println("Invalid input path to source file.");
        }
        String[] sourceCodeLines = sourceCode.lines().toArray(String[]::new);

        if (td.getSuperclassType() != null)
            superClassName = FQNSequenceGenerator.getUnresolvedType(td.getSuperclassType());
        for (MethodDeclaration method : td.getMethods()) {
            int methodStartLine = ast.getLineNumber(method.getStartPosition()) - 1;
//            int methodEndLine;
//            if (sourceCodeLines[methodStartLine].contains("@")) {
//                methodEndLine = methodStartLine + (int) method.toString().lines().count() + 1;
//            } else {
//                methodEndLine = methodStartLine + (int) method.toString().lines().count();
//            }
            int methodEndLine = ast.getLineNumber(method.getStartPosition() + method.getLength()) - 1;
            List<String> methodLines = IntStream.range(0, sourceCodeLines.length)
                                                .filter(i -> (i >= methodStartLine && i <= methodEndLine))
                                                .mapToObj(i -> sourceCodeLines[i])
                                                .collect(Collectors.toList());
            String methodSourceCode = String.join("\n", methodLines);
            int methodStartOffset = method.getStartPosition() - 1;
            boolean correctTrailingNewLines = false;
            if (lib == "com.google.gwt")
                correctTrailingNewLines = true;
            FQNSequenceGenerator sg = new FQNSequenceGenerator(className, superClassName, methodStartOffset, methodStartLine, correctTrailingNewLines, ast);
            method.accept(sg);
			numOfSequences++;

            String outputFile = name + "." + buildSignature(method);
            ArrayList<HashMap<String, String>> nodeInfo = sg.getNodeInfo();
            // Save to file.
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("pathToSource", path);
            jsonObject.put("methodSnippet", methodSourceCode);
            jsonObject.put("nodeInfo", nodeInfo);

            try {
                FileWriter outputJsonFile = new FileWriter(this.outPath + "/" + outputFile + ".json");
                outputJsonFile.write(jsonObject.toString());
                outputJsonFile.close();
            } catch (IOException ex) {
                System.out.println("Invalid output path to node outputs.");
            }
        }
        for (TypeDeclaration inner : td.getTypes())
            numOfSequences += generateSequence(ast, keepUnresolvables, lib, inner, path, name);
        return numOfSequences;
    }

    private String[] getSourcePaths(String path, String[] extensions) {
        HashSet<String> exts = new HashSet<>();
        for (String e : extensions)
            exts.add(e);
        HashSet<String> paths = new HashSet<>();
        getSourcePaths(new File(path), paths, exts);
        paths.removeAll(badFiles);
        return (String[]) paths.toArray(new String[0]);
    }

    private void getSourcePaths(File file, HashSet<String> paths, HashSet<String> exts) {
        if (file.isDirectory()) {
            for (File sub : file.listFiles())
                getSourcePaths(sub, paths, exts);
        } else if (exts.contains(getExtension(file.getName())))
            paths.add(file.getAbsolutePath());
    }

    private Object getExtension(String name) {
        int index = name.lastIndexOf('.');
        if (index < 0)
            index = 0;
        return name.substring(index);
    }

    private String[] getJarPaths() {
        HashMap<String, File> jarFiles = new HashMap<>();
        HashSet<String> globalRepoLinks = new HashSet<>();
        globalRepoLinks.add("http://central.maven.org/maven2/");
        HashMap<String, String> globalProperties = new HashMap<>();
        HashMap<String, String> globalManagedDependencies = new HashMap<>();
        Stack<ClassPathUtil.PomFile> parentPomFiles = new Stack<>();
        getJarFiles(new File(inPath), jarFiles, globalRepoLinks, globalProperties, globalManagedDependencies, parentPomFiles);
        String[] paths = new String[jarFiles.size()];
        int i = 0;
        for (File file : jarFiles.values())
            paths[i++] = file.getAbsolutePath();
        return paths;
    }

    private String[] getJarPaths(String libFolderPath){
        File directoryPath = new File(libFolderPath);
		String[] extensions = new String[] {"jar"};
		List<File> filesList = (List<File>) FileUtils.listFiles(directoryPath, extensions, true);
        String[] jarPaths = new String[filesList.size()];
        for(int i = 0; i < filesList.size(); i++) {
            jarPaths[i] = filesList.get(i).toString();
        }
        return jarPaths;
    }

    private void getJarFiles(File file, HashMap<String, File> jarFiles, 
            HashSet<String> globalRepoLinks, HashMap<String, String> globalProperties, HashMap<String, String> globalManagedDependencies,
            Stack<PomFile> parentPomFiles) {
        if (file.isDirectory()) {
            int size = parentPomFiles.size();
            ArrayList<File> dirs = new ArrayList<>();
            for (File sub : file.listFiles()) {
                if (sub.isDirectory())
                    dirs.add(sub);
                else
                    getJarFiles(sub, jarFiles, globalRepoLinks, globalProperties, globalManagedDependencies, parentPomFiles);
            }
            for (File dir : dirs)
                getJarFiles(dir, jarFiles, globalRepoLinks, globalProperties, globalManagedDependencies, parentPomFiles);
            if (parentPomFiles.size() > size)
                parentPomFiles.pop();
        } else if (file.getName().endsWith(".jar")) {
            File f = jarFiles.get(file.getName());
            if (f == null || file.lastModified() > f.lastModified())
                jarFiles.put(file.getName(), file);
        } else if (file.getName().equals("build.gradle")) {
            try {
                ClassPathUtil.getGradleDependencies(file, this.inPath + "/lib");
            } catch (Throwable t) {
                t.printStackTrace();
            }
        } else if (file.getName().equals("pom.xml")) {
            try {
                ClassPathUtil.getPomDependencies(file, this.inPath + "/lib", globalRepoLinks, globalProperties, globalManagedDependencies, parentPomFiles);
            } catch (Throwable t) {
                t.printStackTrace();
            }
        }
    }

    public static String buildSignature(MethodDeclaration method) {
        StringBuilder sb = new StringBuilder();
        sb.append(method.getName().getIdentifier() + "#");
        for (int i = 0; i < method.parameters().size(); i++) {
            SingleVariableDeclaration svd = (SingleVariableDeclaration) method.parameters().get(i);
            sb.append(getSimpleType(svd.getType()) + "#");
        }
        return sb.toString();
    }

    public static String getSimpleType(VariableDeclarationFragment f) {
        String dimensions = "";
        for (int i = 0; i < f.getExtraDimensions(); i++)
            dimensions += "[]";
        ASTNode p = f.getParent();
        if (p instanceof FieldDeclaration)
            return getSimpleType(((FieldDeclaration) p).getType()) + dimensions;
        if (p instanceof VariableDeclarationStatement)
            return getSimpleType(((VariableDeclarationStatement) p).getType()) + dimensions;
        if (p instanceof VariableDeclarationExpression)
            return getSimpleType(((VariableDeclarationExpression) p).getType()) + dimensions;
        throw new UnsupportedOperationException("Get type of a declaration!!!");
    }

    public static String getSimpleType(Type type) {
        if (type.isArrayType()) {
            ArrayType t = (ArrayType) type;
            String pt = getSimpleType(t.getElementType());
            for (int i = 0; i < t.getDimensions(); i++)
                pt += "[]";
            return pt;
            //return type.toString();
        } else if (type.isParameterizedType()) {
            ParameterizedType t = (ParameterizedType) type;
            return getSimpleType(t.getType());
        } else if (type.isPrimitiveType()) {
            String pt = type.toString();
            /*if (pt.equals("byte") || pt.equals("short") || pt.equals("int") || pt.equals("long") 
                    || pt.equals("float") || pt.equals("double"))
                return "number";*/
            return pt;
        } else if (type.isQualifiedType()) {
            QualifiedType t = (QualifiedType) type;
            return t.getName().getIdentifier();
        } else if (type.isSimpleType()) {
            SimpleType st = (SimpleType) type;
            String pt = st.getName().getFullyQualifiedName();
            if (st.getName() instanceof QualifiedName)
                pt = getSimpleName(st.getName());
            if (pt.isEmpty())
                pt = st.getName().getFullyQualifiedName();
            /*if (pt.equals("Byte") || pt.equals("Short") || pt.equals("Integer") || pt.equals("Long") 
                    || pt.equals("Float") || pt.equals("Double"))
                return "number";*/
            return pt;
        } else if (type.isIntersectionType()) {
            IntersectionType it = (IntersectionType) type;
            @SuppressWarnings("unchecked")
            ArrayList<Type> types = new ArrayList<>(it.types());
            String s = getSimpleType(types.get(0));
            for (int i = 1; i < types.size(); i++)
                s += "&" + getSimpleType(types.get(i));
            return s;
        }  else if (type.isUnionType()) {
            UnionType ut = (UnionType) type;
            String s = getSimpleType((Type) ut.types().get(0));
            for (int i = 1; i < ut.types().size(); i++)
                s += "|" + getSimpleType((Type) ut.types().get(i));
            return s;
        } else if (type.isWildcardType()) {
            WildcardType t = (WildcardType) type;
            return getSimpleType(t.getBound());
        } else if (type.isNameQualifiedType()) {
            NameQualifiedType nqt = (NameQualifiedType) type;
            return nqt.getName().getIdentifier();
        } else if (type.isAnnotatable()) {
            return type.toString();
        }
        System.err.println("ERROR: Declare a variable with unknown type!!!");
        System.exit(0);
        return null;
    }

    public static String getSimpleType(Type type, HashSet<String> typeParameters) {
        if (type.isArrayType()) {
            ArrayType t = (ArrayType) type;
            String pt = getSimpleType(t.getElementType(), typeParameters);
            for (int i = 0; i < t.getDimensions(); i++)
                pt += "[]";
            return pt;
            //return type.toString();
        } else if (type.isParameterizedType()) {
            ParameterizedType t = (ParameterizedType) type;
            return getSimpleType(t.getType(), typeParameters);
        } else if (type.isPrimitiveType()) {
            return type.toString();
        } else if (type.isQualifiedType()) {
            QualifiedType t = (QualifiedType) type;
            return t.getName().getIdentifier();
        } else if (type.isSimpleType()) {
            if (typeParameters.contains(type.toString()))
                return "Object";
            return type.toString();
        } else if (type.isIntersectionType()) {
            IntersectionType it = (IntersectionType) type;
            @SuppressWarnings("unchecked")
            ArrayList<Type> types = new ArrayList<>(it.types());
            String s = getSimpleType(types.get(0), typeParameters);
            for (int i = 1; i < types.size(); i++)
                s += "&" + getSimpleType(types.get(i), typeParameters);
            return s;
        } else if (type.isUnionType()) {
            UnionType ut = (UnionType) type;
            String s = getSimpleType((Type) ut.types().get(0), typeParameters);
            for (int i = 1; i < ut.types().size(); i++)
                s += "|" + getSimpleType((Type) ut.types().get(i), typeParameters);
            return s;
        } else if (type.isWildcardType()) {
            WildcardType t = (WildcardType) type;
            return getSimpleType(t.getBound(), typeParameters);
        } else if (type.isNameQualifiedType()) {
            NameQualifiedType nqt = (NameQualifiedType) type;
            return nqt.getName().getIdentifier();
        } else if (type.isAnnotatable()) {
            return type.toString();
        }
        System.err.println("ERROR: Declare a variable with unknown type!!!");
        System.exit(0);
        return null;
    }

    public static String getSimpleName(Name name) {
        if (name.isSimpleName()) {
            SimpleName sn = (SimpleName) name;
            if (Character.isUpperCase(sn.getIdentifier().charAt(0)))
                return sn.getIdentifier();
            return "";
        }
        QualifiedName qn = (QualifiedName) name;
        if (Character.isUpperCase(qn.getFullyQualifiedName().charAt(0)))
            return qn.getFullyQualifiedName();
        String sqn = getSimpleName(qn.getQualifier());
        if (sqn.isEmpty())
            return getSimpleName(qn.getName());
        return sqn + "." + qn.getName().getIdentifier();
    }
}
