unit Unit1;

{$mode Delphi}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, Generics.Collections,
  StdCtrls, Math, StrUtils, Windows;

const
  LLAMA_DLL = 'llama.dll';
  GGML_DLL = 'ggml.dll';

type
  TFormLlamaTest = class(TForm)
    BtnLoadModel: TButton;
    EditModelPath: TEdit;
    MemoLog: TMemo;
    OpenDialog1: TOpenDialog;
    procedure BtnLoadModelClick(Sender: TObject);
  private
    procedure Log(const Msg: string);
  public
    procedure RunSimpleChat(const ModelPath: string);
  end;

  ByteBool = Byte;
  PUTF8Char = PAnsiChar;

  llama_progress_callback = function(progress: Single; user_data: Pointer): Boolean; cdecl;

  llama_model_params = record
    devices: Pointer;
    tensor_buft_overrides: Pointer;
    n_gpu_layers: Int32;
    split_mode: Int32;
    main_gpu: Int32;
    tensor_split: PSingle;
    progress_callback: llama_progress_callback;
    progress_callback_user_data: Pointer;
    kv_overrides: Pointer;
    vocab_only: ByteBool;
    use_mmap: ByteBool;
    use_mlock: ByteBool;
    check_tensors: ByteBool;
    cont_batching: ByteBool;
    padding1: ByteBool;
    padding2: Int32;
    batch_size_suggestion: Int32;
    model_path: PUTF8Char;
  end;

  Pllama_model = Pointer;

var
  llama_model_load_from_file: function(const path_model: PUTF8Char; params: llama_model_params): Pllama_model; cdecl;
  llama_model_default_params: function: llama_model_params; cdecl;
  llama_backend_init: procedure; cdecl;
  ggml_backend_load_all: procedure; cdecl;
  FormLlamaTest: TFormLlamaTest;

function LoadLlamaDLL(const BasePath: string): Boolean;
function MyProgressCallback(progress: Single; user_data: Pointer): Boolean; cdecl;

implementation

{$R *.lfm}

var
  LlmHandle: HMODULE = 0;
  GgmlHandle: HMODULE = 0;

function LoadLlamaDLL(const BasePath: string): Boolean;
var
  llamaPath, ggmlPath: string;
begin
  Result := False;

  llamaPath := IncludeTrailingPathDelimiter(BasePath) + LLAMA_DLL;
  ggmlPath := IncludeTrailingPathDelimiter(BasePath) + GGML_DLL;

  LlmHandle := LoadLibrary(PChar(llamaPath));
  if LlmHandle = 0 then Exit;

  GgmlHandle := LoadLibrary(PChar(ggmlPath));
  if GgmlHandle = 0 then Exit;

  @llama_model_load_from_file := GetProcAddress(LlmHandle, 'llama_model_load_from_file');
  @llama_model_default_params := GetProcAddress(LlmHandle, 'llama_model_default_params');
  @llama_backend_init := GetProcAddress(LlmHandle, 'llama_backend_init');
  @ggml_backend_load_all := GetProcAddress(GgmlHandle, 'ggml_backend_load_all');

  Result := Assigned(llama_model_load_from_file) and
            Assigned(llama_model_default_params) and
            Assigned(llama_backend_init) and
            Assigned(ggml_backend_load_all);
end;

function MyProgressCallback(progress: Single; user_data: Pointer): Boolean; cdecl;
begin
  TThread.Queue(nil,
    procedure
    begin
      if Assigned(FormLlamaTest) and Assigned(FormLlamaTest.MemoLog) then
        FormLlamaTest.MemoLog.Lines.Add(Format('Model loading: %.1f%%', [progress * 100]));
    end);
  Result := True;
end;

procedure TFormLlamaTest.Log(const Msg: string);
begin
  MemoLog.Lines.Add(Msg);
end;

procedure TFormLlamaTest.RunSimpleChat(const ModelPath: string);
var
  model_params: llama_model_params;
  FLlmModel: Pllama_model;
  UTF8ModelPath: UTF8String;
  OldMask: TFPUExceptionMask;
  DllFolder: string;
begin
  DllFolder := ExtractFilePath(ParamStr(0));
  Log('DLL folder: ' + DllFolder);
  if not LoadLlamaDLL(DllFolder) then
  begin
    Log('Failed to load required DLLs or bind functions.');
    Exit;
  end;

  OldMask := GetExceptionMask;
  SetExceptionMask(OldMask + [exInvalidOp, exOverflow, exZeroDivide]);

  Log('Calling ggml_backend_load_all...');
  try
    if Assigned(ggml_backend_load_all) then
    begin
      ggml_backend_load_all();
      Log('ggml_backend_load_all executed');
    end;
  except
    on e: Exception do
      Log('Exception in ggml_backend_load_all: ' + e.Message);
  end;

  Log('Calling llama_backend_init...');
  llama_backend_init;

  try
    model_params := llama_model_default_params;
    model_params.n_gpu_layers := 9999;
    model_params.split_mode := 0;
    model_params.main_gpu := 0;
    model_params.use_mmap := ByteBool(1);
    model_params.use_mlock := ByteBool(0);
    model_params.check_tensors := ByteBool(0);
    model_params.vocab_only := ByteBool(0);
    model_params.progress_callback := @MyProgressCallback;
    model_params.progress_callback_user_data := nil;
    model_params.model_path := nil;
    model_params.devices := nil;
    model_params.tensor_buft_overrides := nil;
    model_params.tensor_split := nil;
    model_params.kv_overrides := nil;
    model_params.cont_batching := ByteBool(0);
    model_params.padding1 := ByteBool(0);
    model_params.padding2 := 0;
    model_params.batch_size_suggestion := 0;

    UTF8ModelPath := UTF8Encode(ModelPath);
    Log('Loading model: ' + ModelPath);

    try
      FLlmModel := llama_model_load_from_file(PUTF8Char(UTF8ModelPath), model_params);
      if FLlmModel = nil then
        raise Exception.Create('Model load failed.');
      Log('Model loaded successfully!');
    except
      on E: Exception do
        Log('Error during model load: ' + E.ClassName + ' - ' + E.Message);
    end;

  finally
    SetExceptionMask(OldMask);
  end;
end;

procedure TFormLlamaTest.BtnLoadModelClick(Sender: TObject);
begin
  if OpenDialog1.Execute then
  begin
    EditModelPath.Text := OpenDialog1.FileName;
    RunSimpleChat(EditModelPath.Text);
  end;
end;

end.

