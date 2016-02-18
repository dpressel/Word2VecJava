package word2vec.lite;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.After;
import org.junit.Test;

import word2vec.lite.util.Common;

/**
 * Tests converting the binary models into
 * {@link Word2VecModel}s.
 * 
 * @see Word2VecModel#fromBinFile(File)
 * @see Word2VecModel#fromBinFile(File,
 *      java.nio.ByteOrder)
 */
public class Word2VecBinTest {

  /**
   * Tests that the Word2VecModels created from a binary and text
   * representations are equivalent
   */
  @Test
  public void testRead()
      throws IOException, Searcher.UnknownWordException
  {
    File binFile = Common.getResourceAsFile(
            this.getClass(),
            "/word2vec/lite/tokensModel.bin");
    Word2VecModel binModel = Word2VecModel.fromBinFile(binFile);

  }

  private Path tempFile = null;

  @After
  public void cleanupTempFile() throws IOException {
    if(tempFile != null)
      Files.delete(tempFile);
  }

}
